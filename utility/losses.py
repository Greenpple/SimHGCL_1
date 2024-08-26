import torch


def get_bpr_loss(user_embedding, positive_embedding, negative_embedding):
    #: 计算用户嵌入向量与正样本嵌入向量的内积，表示用户与正样本之间的相似度得分。
    pos_score = torch.sum(torch.mul(user_embedding, positive_embedding), dim=1)
    #: 计算用户嵌入向量与负样本嵌入向量的内积，表示用户与负样本之间的相似度得分。
    neg_score = torch.sum(torch.mul(user_embedding, negative_embedding), dim=1)

    # loss = torch.mean(torch.nn.functional.softplus(neg_score - pos_score))
    #计算BPR损失，首先通过sigmoid函数将正样本得分与负样本得分的差异转换为0到1之间的值，然后取其自然对数。
    # 添加一个小的常数（10e-8）是为了防止取对数时出现0值。
    loss = - torch.log(torch.sigmoid(pos_score - neg_score) + 10e-8)

    return torch.mean(loss)


def get_reg_loss(*embeddings):
    reg_loss = 0
    for embedding in embeddings:
        reg_loss += 1 / 2 * embedding.norm(2).pow(2) / float(embedding.shape[0])

    return reg_loss


def get_InfoNCE_loss(embedding_1, embedding_2, temperature):
    #对输入的第一个嵌入向量进行归一化处理，确保其单位长度。
    embedding_1 = torch.nn.functional.normalize(embedding_1)
    #对输入的第二个嵌入向量进行归一化处理，确保其单位长度。
    embedding_2 = torch.nn.functional.normalize(embedding_2)

    #计算两个归一化后的嵌入向量的点积，并在最后一个维度上求和，得到正样本的得分。
    pos_score = (embedding_1 * embedding_2).sum(dim=-1)
    #将正样本得分除以温度参数后取指数，用于增加得分的区分度。
    pos_score = torch.exp(pos_score / temperature)

    #计算两个嵌入向量的内积，得到总体得分，这里采用了矩阵乘法。
    ttl_score = torch.matmul(embedding_1, embedding_2.transpose(0, 1))
    #将总体得分除以温度参数后取指数，然后在第一个维度上求和，得到总体得分的加权和。
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)

    #计算InfoNCE损失，首先将正样本得分除以总体得分的加权和，然后取对数，并取负值。
    cl_loss = - torch.log(pos_score / ttl_score + 10e-6)
    #返回InfoNCE损失的平均值，即对整个batch的损失进行了平均。
    return torch.mean(cl_loss)


def get_ELBO_loss(recon_x, x, mu, logvar, anneal):
    BCE = - torch.mean(torch.sum(torch.nn.functional.log_softmax(recon_x, 1) * x, -1))

    KLD = - 0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return BCE + anneal * KLD


def get_align_loss(embedding_1, embedding_2):
    embedding_1 = torch.nn.functional.normalize(embedding_1, dim=-1)
    embedding_2 = torch.nn.functional.normalize(embedding_2, dim=-1)
    return torch.mean((embedding_1 - embedding_2).norm(p=2, dim=1).pow(2))
   
   
def get_transR_loss(embedding_h, embedding_t, embedding_r):
    embedding_h = torch.nn.functional.normalize(embedding_h, dim=-1)
    embedding_t = torch.nn.functional.normalize(embedding_t, dim=-1)
    embedding_r = torch.nn.functional.normalize(embedding_r, dim=-1)
    return torch.mean((embedding_h + embedding_r - embedding_t).norm(p=2, dim=1).pow(2))


def get_uniform_loss(embedding):
    embedding = torch.nn.functional.normalize(embedding, dim=-1)
    return torch.pdist(embedding, p=2).pow(2).mul(-2).exp().mean().log()

