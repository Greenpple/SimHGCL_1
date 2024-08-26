import utility.losses
import utility.tools
import utility.trainer
import torch
from torch import nn
import utility.losses
import utility.tools
import utility.trainer

class GCRec(nn.Module):
    def __init__(self, config, dataset, user_g, item_g, device):
        super(GCRec, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config.reg_lambda)
        self.ssl_lambda = float(self.config.ssl_lambda)
        self.temperature = float(self.config.temperature)
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users, embedding_dim=int(self.config.dim))
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items, embedding_dim=int(self.config.dim))

        # no pretrain
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)
        self.Graph = []
        for i in range(len(user_g)):
            graph = self.dataset.sparse_adjacency_matrix()
            #graph[:self.dataset.num_users, :self.dataset.num_users] = user_g[i]
            #graph[self.dataset.num_users:, self.dataset.num_users:] = item_g[i]
            graph_tensor = utility.tools.convert_sp_mat_to_sp_tensor(graph)
            graph_tensor_coalesced = graph_tensor.coalesce().to(self.device)
            self.Graph.append(graph_tensor_coalesced)
        self.activation = nn.Sigmoid()

    def aggregate(self, id):
        # [user + item, emb_dim]
        all_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight])

        # no dropout
        embeddings = []
        # all_embedding = self.mask(all_embedding)
        for layer in range(int(self.config.GCN_layer)):
            all_embedding = torch.sparse.mm(self.Graph[id], all_embedding)
            embeddings.append(all_embedding)

        final_embeddings = torch.stack(embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return users_emb, items_emb

    def forward(self, user, positive, negative, epoch=None):
        user_embedding_1, item_embedding_1 = self.aggregate(0)
        user_embedding_2, item_embedding_2 = self.aggregate(1)

        all_user_embeddings = (user_embedding_1 + user_embedding_2) / 2
        all_item_embeddings = (item_embedding_1 + item_embedding_2) / 2

        user_embedding = all_user_embeddings[user.long()]
        pos_embedding = all_item_embeddings[positive.long()]
        neg_embedding = all_item_embeddings[negative.long()]

        ego_user_emb = self.user_embedding(user)
        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        bpr_loss = utility.losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)

        reg_loss = utility.losses.get_reg_loss(ego_user_emb, ego_pos_emb, ego_neg_emb)
        reg_loss = self.reg_lambda * reg_loss

        user_ssl_loss = utility.losses.get_InfoNCE_loss(user_embedding_1[user.long()],
                                                        user_embedding_2[user.long()],
                                                        self.temperature)
        item_ssl_loss = utility.losses.get_InfoNCE_loss(item_embedding_1[positive.long()],
                                                        item_embedding_2[positive.long()],
                                                        self.temperature)

        ssl_loss = self.ssl_lambda * (user_ssl_loss + item_ssl_loss)

        loss_list = [bpr_loss, reg_loss, ssl_loss]

        return loss_list


    def get_rating_for_test(self, user):
        all_user_embeddings, all_item_embeddings = self.aggregate(0)
        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating

    def get_embedding(self):
        all_user_embeddings, all_item_embeddings = self.aggregate(0)
        return all_user_embeddings, all_item_embeddings


    def mask(self, x, mask_rate=0.3):
        perm = torch.randperm(self.dataset.num_users, device=x.device)
        # random masking
        num_mask_nodes = int(mask_rate * self.dataset.num_users)
        mask_nodes = perm[: num_mask_nodes]


        out_x = x.clone()
        out_x[mask_nodes] = 0.0
        return out_x