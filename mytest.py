import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from collections import Counter

# 1. 读取词频数据
def read_word_freq(file_path, keywords):
    word_freq = {word: 0 for word in keywords}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, freq = line.strip().split()
            if word in word_freq:
                word_freq[word] = int(freq)
    return list(word_freq.values())

keywords = ['之', '其', '或', '亦', '方', '于', '即', '皆', '因', '仍','故', '尚', '呢', '了', '的', '着', '一', '不', '乃', '呀','吗', '咧', '啊', '把', '让', '向', '往', '是', '在', '越','再', '更', '比', '很', '偏', '别', '好', '可', '便', '就','但', '儿','又', '也', '都', '要','这', '那', '你', '我', '他','来', '去', '道', '笑', '说']
keywords = ["了", "的", "道", "也", "他", "是", "我", "你", "着", "又", "说", "来", "不"]

files = os.listdir('mywc')

# 读取所有章节的词频数据
word_vectors = []
for file_path in files:
    if not file_path.startswith('dd'):
        word_vectors.append(read_word_freq(f'mywc/{file_path}', keywords))

# 转换为矩阵
X = np.array(word_vectors)

# 2. 计算距离
distances = euclidean_distances(X)
# distances = cosine_distances(X)

# 3. 使用K-means聚类（簇数分别为2和3）
def kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=39)
    labels = kmeans.fit_predict(X)
    return labels, kmeans

def calculate_proportions(labels):
    # 假设文件划分为前40个、第41-120个、第121-148个章节
    # 对应于0-39, 40-119, 120-147
    ranges = [(0, 39), (40, 119), (120, 147)]  # 注意索引从0开始
    proportions = {}

    for i, (start, end) in enumerate(ranges):
        range_labels = labels[start:end+1]  # 截取对应区间的标签
        label_counts = Counter(range_labels)
        total_count = len(range_labels)
        
        # 计算比例
        proportions[i] = {label: count / total_count for label, count in label_counts.items()}
    
    return proportions

# 4. 聚类并展示结果
for n_clusters in [2, 3]:
    labels, kmeans = kmeans_clustering(X, n_clusters)
    print(f"\nK-means 聚类结果 (簇数={n_clusters}):")
    print(labels)
    
    # 打印每个区间的比例
    proportions = calculate_proportions(labels)
    for i, prop in proportions.items():
        start, end = [(0, 39), (40, 119), (120, 147)][i]
        print(f"\n区间 {start+1}-{end+1}:")
        for label, proportion in prop.items():
            print(f"  标签 {label}: {proportion:.2f}")

    # 可视化聚类结果
    plt.scatter(range(len(labels)), labels, c=labels, cmap='viridis')
    plt.title(f'K-means Clustering Results (n_clusters={n_clusters})')
    plt.xlabel('Chapter Index')
    plt.ylabel('Cluster Label')
    plt.show()
