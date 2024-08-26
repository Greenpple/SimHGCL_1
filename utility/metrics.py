"""
Created on April 18, 2021
PyTorch Implementation of GNN-based Recommender System
This file is used to evaluate the performance of the model(e.g. recall, ndcg, precision, hit)
"""
import numpy as np


def ndcg_at_k(r, k, test_data):
    """
        Normalized discounted cumulative gain
    """
    #使用断言确保预测结果和测试数据长度一致。
    assert len(r) == len(test_data)   # r：模型预测结果 K：前K个推荐物品 test_data:每个元素都是一个列表

    prediction_data = r[:, :k]        # 从预测结果中提取前K个推荐物品的数据
    test_matrix = np.zeros((len(prediction_data), k)) # 用于表示测试数据的二值化矩阵
    for i, items in enumerate(test_data): # 遍历测试数据中的每个用户的数据。
        length = k if k <= len(items) else len(items) # 确定当前用户测试数据中要考虑的物品数量，取 k 和用户实际数据数量的最小值。
        test_matrix[i, :length] = 1 # 将测试矩阵中对应位置设为1，表示该物品在测试数据中存在。

    max_r = test_matrix # 将二值化的测试矩阵赋值给 max_r，用于计算理想的折损累积增益。
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1) # 计算理想情况下的折损累积增益（IDCG）。
    dcg = np.sum(prediction_data * (1. / np.log2(np.arange(2, k + 2))), axis=1) # 计算模型预测结果的折损累积增益（DCG）。
    idcg[idcg == 0.] = 1. # 将 IDCG 中为零的值设置为1，以避免除以零的情况。
    ndcg = dcg / idcg # 计算归一化折损累积增益（NDCG）。
    ndcg[np.isnan(ndcg)] = 0. # 返回 NDCG 的总和作为结果。
    return np.sum(ndcg) # 返回 NDCG 的总和作为结果。


def recall_at_k(r, k, test_data):
    right_prediction = r[:, :k].sum(1) # 对预测结果中前 k 个推荐物品的每一行进行求和，得到每个用户的正确预测数量。
    recall_num = np.array([len(test_data[i]) for i in range(len(test_data))]) #计算每个用户测试数据中的物品数量，构成一个数组。
    recall = np.sum(right_prediction / recall_num) # 即正确预测的总数量除以测试数据中物品的总数量，得到召回率。
    return recall


def precision_at_k(r, k, test_data):
    right_prediction = r[:, :k].sum(1)
    precision_num = k
    precision = np.sum(right_prediction) / precision_num
    return precision


def F1(pre, rec):
    F1 = []
    for i in range(len(pre)):
        if pre[i] + rec[i] > 0:
            F1.append((2.0 * pre[i] * rec[i]) / (pre[i] + rec[i]))
        else:
            F1.append(0.)
    return F1


def get_label(true_data, pred_data):
    r = []
    for i in range(len(true_data)):
        ground_true = true_data[i]
        pred_top_k = pred_data[i]
        pred = list(map(lambda x: x in ground_true, pred_top_k))
        pred = np.array(pred).astype("float")
        r.append(pred)

    return np.array(r).astype("float")
