import scipy.sparse as sp
import numpy as np

def load_data(dataset, col_D_inv):
    if dataset == 'douban-movie':
        # uuuuu_graph = sp.load_npz('data/douban-movie/uuuuu.npz')
        # umamu_graph = sp.load_npz('data/douban-movie/umamu.npz')
        # uumuu_graph = sp.load_npz('data/douban-movie/uumuu.npz')
        umu_graph = sp.load_npz('data/douban-movie/umu_processed.npz')
        ugu_graph = sp.load_npz('data/douban-movie/ugu.npz')
        # mam_graph = sp.load_npz('data/douban-movie/mam_processed.npz')
        mum_graph = sp.load_npz('data/douban-movie/mum_processed.npz')
        # mtm_graph = sp.load_npz('data/douban-movie/mtm.npz')
        mdm_graph = sp.load_npz('data/douban-movie/mdm.npz')

        umu_graph = coo_To_csr(umu_graph, col_D_inv)
        ugu_graph = coo_To_csr(ugu_graph, col_D_inv)
        mum_graph = coo_To_csr(mum_graph, col_D_inv)
        mdm_graph = coo_To_csr(mdm_graph, col_D_inv)

        user_mps, item_mps = [umu_graph, ugu_graph], [mum_graph, mdm_graph]

    elif dataset == 'yelp':
        # uu_graph = sp.load_npz('data/yelp/uu.npz')
        ubu_graph = sp.load_npz('data/yelp/ubu.npz')
        ucu_graph = sp.load_npz('data/yelp/ucu.npz')
        # bub_graph = sp.load_npz('data/yelp/bub.npz')
        bcab_graph = sp.load_npz('data/yelp/bcab.npz')
        bcib_graph = sp.load_npz('data/yelp/bcib.npz')

        ucu_graph = coo_To_csr(ucu_graph, col_D_inv)
        ubu_graph = coo_To_csr(ubu_graph, col_D_inv)
        bcab_graph = coo_To_csr(bcab_graph, col_D_inv)
        bcib_graph = coo_To_csr(bcib_graph, col_D_inv)

        user_mps, item_mps = [ucu_graph, ubu_graph], [bcib_graph, bcab_graph] # , ubu_graph, bcib_graph

    elif dataset == 'amazon':
        uibiu_graph = sp.load_npz('data/amazon/uibiu_processed.npz')
        uiu_graph = sp.load_npz('data/amazon/uiu_processed.npz')
        # ibi_graph = sp.load_npz('data/amazon/ibi.npz')
        iui_graph = sp.load_npz('data/amazon/iui.npz')
        # iui_graph = sp.load_npz('data/amazon/iui.npz')
        ici_graph = sp.load_npz('data/amazon/ici_processed.npz')

        uibiu_graph = coo_To_csr(uibiu_graph, col_D_inv)
        uiu_graph = coo_To_csr(uiu_graph, col_D_inv)
        iui_graph = coo_To_csr(iui_graph, col_D_inv)
        ici_graph = coo_To_csr(ici_graph, col_D_inv)

        user_mps, item_mps = [uibiu_graph, uiu_graph], [iui_graph, ici_graph]  #

    elif dataset == 'movielens-1m':
        # umgmu_graph = sp.load_npz('data/movielens-1m/umgmu.npz')
        # uu_graph = sp.load_npz('data/movielens-1m/uu.npz')
        uou_graph = sp.load_npz('data/movielens-1m/uou_processed.npz')
        # uau_graph = sp.load_npz('data/movielens-1m/uau.npz')
        umu_graph = sp.load_npz('data/movielens-1m/umu_processed.npz')
        # uumuu_graph = sp.load_npz('data/movielens-1m/uumuu.npz')
        # mm_graph = sp.load_npz('data/movielens-1m/mm.npz')
        # mmumm_graph = sp.load_npz('data/movielens-1m/mmumm.npz')
        mum_graph = sp.load_npz('data/movielens-1m/mum_processed.npz')
        mgm_graph = sp.load_npz('data/movielens-1m/mgm_processed.npz')

        umu_graph = coo_To_csr(umu_graph, col_D_inv)
        uou_graph = coo_To_csr(uou_graph, col_D_inv)
        mum_graph = coo_To_csr(mum_graph, col_D_inv)
        mgm_graph = coo_To_csr(mgm_graph, col_D_inv)

        user_mps, item_mps = [uou_graph, umu_graph], [mgm_graph, mum_graph]  #

    return user_mps, item_mps

def coo_To_csr(coo_matrix, col_D_inv):
    coo_matrix = coo_matrix + sp.eye(coo_matrix.shape[0])
    csr = coo_matrix.tocsr()
    col_sum = np.array(csr.sum(axis=0)).flatten()
    col_d_inv = np.power(col_sum, col_D_inv)
    col_d_inv[np.isinf(col_d_inv)] = 0.
    csr_d_matrix = sp.diags(col_d_inv)
    norm_adjacency = csr_d_matrix.dot(csr_d_matrix).dot(csr_d_matrix)

    return norm_adjacency