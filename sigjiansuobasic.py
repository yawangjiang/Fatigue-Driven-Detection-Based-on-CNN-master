import numpy as np
import os

import torch
from sklearn.metrics.pairwise import cosine_similarity


def load_features_from_folder(folder_path):
    """
    加载文件夹中所有的.npy特征文件
    返回文件名列表和对应的特征矩阵
    """
    filenames = []
    features = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            filepath = os.path.join(folder_path, filename)
            feature = np.load(filepath)
            # 确保特征形状是(1, 1024)
            if feature.shape == (1280,) or feature.shape == (1, 1280):
                feature = feature.reshape(1, -1)  # 统一为(1, 1024)形状
                filenames.append(filename)
                features.append(feature)

    if not features:
        raise ValueError("文件夹中没有找到有效的.npy特征文件")

    # 将所有特征堆叠成一个矩阵 (n_samples, 1024)
    features_matrix = np.vstack(features)
    return filenames, features_matrix


def find_most_similar(query_feature, folder_path, top_k=1):
    """
    查询最相似的特征

    参数:
        query_feature: 查询特征，形状为(1, 1024)的numpy数组
        folder_path: 存储特征数据库的文件夹路径
        top_k: 返回最相似的k个结果

    返回:
        list: 包含(top_k)个元组的列表，每个元组是(文件名, 相似度)
    """
    # 确保查询特征形状正确
    if torch.is_tensor(query_feature):
        # 确保在CPU上且无梯度
        query_feature = query_feature.cpu().detach().numpy()
    query_feature = query_feature.reshape(1, -1)
    if query_feature.shape != (1, 1280):
        print(query_feature.shape)
        raise ValueError("查询特征形状不一致")

    # 加载数据库特征
    filenames, features_matrix = load_features_from_folder(folder_path)

    # 计算余弦相似度
    similarities = cosine_similarity(query_feature, features_matrix)
    similarities = similarities.flatten()  # 转换为1D数组

    # 获取top_k最相似的索引
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # 构建结果列表
    results = []
    for idx in top_indices:
        results.append((filenames[idx], similarities[idx]))
    print(results)

    return results


# 示例用法
if __name__ == "__main__":
    # 假设这是你的网络输出特征
    query_feature = np.random.rand(1, 1280)  # 替换为你的实际特征

    # 特征数据库文件夹路径
    database_folder = "./dataset/face_features"

    # 查找最相似的特征
    most_similar = find_most_similar(query_feature, database_folder, top_k=1)

    # 打印结果
    for filename, similarity in most_similar:
        print(f"最相似的特征文件: {filename}")
        print(f"余弦相似度: {similarity:.4f}")