# 凝聚层次聚类：AGNES算法(自底向上)
# 首先将每个对象作为一个簇，然后合并这些原子簇为越来越大的簇，直到某个终结条件被满足

from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

iris = datasets.load_iris()
iris_data = iris.data

clustering = AgglomerativeClustering(linkage='ward', n_clusters=3)
result = clustering.fit(iris_data)

print("各类别的样本数目：")
print(pd.Series(clustering.labels_).value_counts())
print("聚类结果：")
print(confusion_matrix(clustering.labels_, iris.target))

plt.figure()
d0 = iris_data[clustering.labels_ == 0]
plt.scatter(d0[:, 0], d0[:, 1])
d1 = iris_data[clustering.labels_ == 1]
plt.scatter(d1[:, 0], d1[:, 1])
d2 = iris_data[clustering.labels_ == 2]
plt.scatter(d2[:, 0], d2[:, 1])
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title("层次聚类自底向上算法聚类AGNES Clustering")
plt.show()

