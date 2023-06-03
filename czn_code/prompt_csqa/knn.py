import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import jsonlines
# 读取数据
def getdata(data_dir):
    data = []
    with jsonlines.open(data_dir,'r') as f:
        for line in f:
            data.append(line)
    return data

# 使用Bert模型编码文本
def encode_search(data_dir):
    with jsonlines.open(data_dir,'r') as f:
        for line in f:
            data.append(line)
model = SentenceTransformer('bert-base-nli-mean-tokens')
embeddings = model.encode(texts)

# 使用KMeans算法进行聚类
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(embeddings)

# 选择与目标输入最相近的k个样例
input_text = "这是一个示例输入"
input_embedding = model.encode([input_text])
k = 5
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(embeddings)
distances, indices = neigh.kneighbors(input_embedding)

# 输出结果
print("输入文本：", input_text)
print("最相似的", k, "个样例：")
for i in indices[0]:
    print(texts[i])
