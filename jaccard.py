import os
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt

# 计算Jaccard相似度
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

# 读取文件并提取前100个高频词
def read_top_n_words(filepath, top_n=100):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        word_set = set()
        for line in lines[:top_n]:  # 只取前100个
            word = line.split()[0]  # 获取词语（不包括频率）
            word_set.add(word)
        return word_set

# 读取所有章节的词频数据并整理成字典
def load_data_from_directory(directory, top_n=100):
    word_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            chapter_name = filename.split('.')[0]
            word_dict[chapter_name] = read_top_n_words(filepath, top_n)
    return word_dict

# 计算相似度矩阵
def calculate_similarity_matrix(word_dict1, word_dict2):
    similarity_matrix = {}
    for chapter1, words1 in word_dict1.items():
        similarity_matrix[chapter1] = {}
        for chapter2, words2 in word_dict2.items():
            similarity_matrix[chapter1][chapter2] = jaccard_similarity(words1, words2)
    return similarity_matrix

# 计算相似度并输出
def calculate_and_print_similarity():
    # 加载曹雪芹、高鹗、癸酉本的词频数据
    cao_data = load_data_from_directory('./mywc/80', top_n=100)  # 曹雪芹数据
    gao_data = load_data_from_directory('./mywc/40', top_n=100)  # 高鹗数据
    gui_data = load_data_from_directory('./mywc/Guiyou', top_n=100)  # 癸酉本数据

    # 计算癸酉本与曹雪芹、高鹗的相似度
    gui_vs_cao_sim = calculate_similarity_matrix(gui_data, cao_data)
    gui_vs_gao_sim = calculate_similarity_matrix(gui_data, gao_data)
    cao_vs_gao_sim = calculate_similarity_matrix(cao_data, gao_data)

    all_similarities = []
    for chapter1 in gui_vs_cao_sim.values():
        for sim_value in chapter1.values():
            all_similarities.append(sim_value)
    mean_similarity = np.mean(all_similarities)
    print(f"Gui vs Cao: {mean_similarity:.4f}")

    all_similarities = []
    for chapter1 in gui_vs_gao_sim.values():
        for sim_value in chapter1.values():
            all_similarities.append(sim_value)
    mean_similarity = np.mean(all_similarities)
    print(f"Gui vs Gao: {mean_similarity:.4f}")

    all_similarities = []
    for chapter1 in cao_vs_gao_sim.values():
        for sim_value in chapter1.values():
            all_similarities.append(sim_value)
    mean_similarity = np.mean(all_similarities)
    print(f"Cao vs Gao: {mean_similarity:.4f}")

# 主函数
if __name__ == '__main__':
    calculate_and_print_similarity()
