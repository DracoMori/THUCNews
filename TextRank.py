'''
date: 2021/2/9
author: 流氓兔23333
content: word2vec + TextRank 进行文本摘要自动提取  and 关键词提取
'''
import numpy as np
import pandas as pd
from tqdm import tqdm  
import time
import random
from string import punctuation
from heapq import nlargest
from itertools import product, count
import math
import os, warnings, pickle
warnings.filterwarnings('ignore')


data_path = 'D:/VSCode/pyStudy/NLP/data/THUCNews/'
save_path = './temp_results/'


import json
class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):                                 
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)

def save_file(filename, dic):
    '''save dict into json file'''
    with open(filename,'w',  encoding='utf-8') as json_file:
        json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)

# 读取js文件
def load_file(filename):
    '''load dict from json file'''
    with open(filename,"r", encoding='utf-8') as json_file:
	    dic = json.load(json_file)
    return dic


# load 原文
dirs_train = str('D:/VSCode/pyStudy/NLP/data/THUCNews/'+'train/')
def read_file(dirs_train, num=1000):
    '''
    return [[文章], [label]]
    '''
    data_train  = [[], []]
    data_val  = [[], []]
    for item in os.listdir(dirs_train):
        print(item, '开始读取')
        new_path = str(dirs_train + item)
        new_dirs = os.listdir(new_path)
        for f_name in new_dirs[:num]:
            file_path = str(new_path+'/'+f_name)
            with open(file_path, encoding='utf-8') as text:
                data_train[0].append(text.read())
                data_train[1].append(item)
    data_train = np.array(data_train)
    return data_train

data_train = read_file(dirs_train)  # [[Text], [label]]
Text = data_train[0][5]

# 按句子分割
sentence_list = Text.split('。')
# 为各个句子去除标点
import re
sentence_list = [[re.sub('\W*', '', t)] for t in sentence_list if len(re.sub('\W*', '', t)) != 0]

# 分词
import jieba 
stop_words = pd.read_table('D:/VSCode/pyStudy/NLP/stopwords/cn_stopwords.txt',header=None)
stop_words = list(stop_words.iloc[1:,0])
sentence_words = [[w for w in jieba.cut(t[0]) if w not in stop_words] for t in sentence_list]


# load分词后的 Text 
# [[text], [label]]
train_df = load_file(data_path+'train_words')


# 训练词向量
import gensim
from gensim.models import word2vec
model = word2vec.Word2Vec(sentences=train_df[0], size=300, window=5, min_count=1, workers=4)
model.save(save_path+'word2vec.model')

train_df[0][0]
model.similarity("马晓旭", "走")


# 获取句子向量
def sentence2vec(sentence_words, word_vectors):
    '''
    返回句子的向量/ 句子向量为 词向量均值
    '''
    sentence_vec = []
    for s in sentence_words:
        words_vec = []
        for w in s:
            words_vec.append(word_vectors[w])
        sentence_vec.append(list(np.array(words_vec).mean(axis=0)))  
    return sentence_vec


''' TextRank 核心 '''
def cosine_similarity(vec1, vec2):
    '''
    计算两个向量之间的余弦相似度
    :param vec1:
    :param vec2:
    :return:
    '''
    tx = np.array(vec1)
    ty = np.array(vec2)
    cos1 = np.sum(tx * ty)
    cos21 = np.sqrt(sum(tx ** 2))
    cos22 = np.sqrt(sum(ty ** 2))
    cosine_value = cos1 / float(cos21 * cos22)
    return cosine_value


def create_graph(word_sent):
    """
    传入句子链表  返回句子之间相似度的图
    :param word_sent:
    :return:
    """
    num = len(word_sent)
    board = [[0.0 for _ in range(num)] for _ in range(num)]
 
    for i, j in product(range(num), repeat=2):
        if i != j:
            board[i][j] = cosine_similarity(word_sent[i], word_sent[j])
    return board
 

def calculate_score(weight_graph, scores, i):
    """
    计算句子在图中的分数
    :param weight_graph:
    :param scores:
    :param i:
    :return:
    """
    length = len(weight_graph)
    d = 0.85
    added_score = 0.0
 
    for j in range(length):
        fraction = 0.0
        denominator = 0.0
        # 计算分子
        fraction = weight_graph[j][i] * scores[j]
        # 计算分母
        for k in range(length):
            denominator += weight_graph[j][k]
            if denominator == 0:
                denominator = 1
        added_score += fraction / denominator
    # 算出最终的分数
    weighted_score = (1 - d) + d * added_score
    return weighted_score


def weight_sentences_rank(weight_graph):
    '''
    输入相似度的图（矩阵)
    返回各个句子的分数
    :param weight_graph:
    :return:
    '''
    # 初始分数设置为0.5
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]
 
    # 开始迭代
    while different(scores, old_scores):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]
        for i in range(len(weight_graph)):
            scores[i] = calculate_score(weight_graph, scores, i)
    return scores
 
 
def different(scores, old_scores):
    '''
    判断前后分数有无变化
    :param scores:
    :param old_scores:
    :return:
    '''
    flag = False
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= 0.0001:
            flag = True
            break
    return flag


# load word2vec model
model = word2vec.Word2Vec.load(save_path+'word2vec.model')

# 获取句子向量
sentence_vec = sentence2vec(sentence_words=sentence_words, word_vectors=model.wv)
np.array(sentence_vec).shape

# 构建图
graph = create_graph(word_sent=sentence_vec)
# 计算得分
scores = weight_sentences_rank(graph)
sen_id = np.array(scores).argmax()
sentence_list[sen_id]
sentence_list[np.array(scores).argmin()]



# ===============  文本关键词提取  ======================
import jieba.analyse as ana
Text = data_train[0][0]
ana.textrank(Text, topK=10, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))



