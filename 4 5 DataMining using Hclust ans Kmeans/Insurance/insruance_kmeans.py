import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
from sklearn.datasets import make_blobs

insurance= pd.read_csv("D:\\360Assignments\\Submission\\DataMining using Hclust ans Kmeans 4 5\\Insurance Dataset.csv")

insurance.describe()
 

def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)


df_norm = norm_func(insurance)

# to know the twss
TWSS = []
k = list(range(1, 5))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

#TWSS 1st cluster has a totalwithinss of 29 which is highest and that means they are well clustered
#Out[19]: [29.645008423361205, 18.56906591852995, 14.489584453634881, 10.796791133859635]

plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")


# elbow curve descents at k=2

model = KMeans(n_clusters = 2)
model.fit(df_norm)
y_kmeans = model.predict(df_norm) # prediction
#each clusters number
model.labels_

#to know the center s of each clusters
model.cluster_centers_

# insruance paid and at what age scatterplot
plt.scatter(df_norm.iloc[:, 0], df_norm.iloc[:,1], c=y_kmeans, s=50, cmap='viridis')

centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
