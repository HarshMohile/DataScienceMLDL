import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
from sklearn.datasets import make_blobs

wine = pd.read_csv("D:\\360Assignments\\Submission\\10 PCA\\wine.csv")
wine.describe()

#To normalize the data 
def norm_func(i):
    x = (i - i.min())/ (i.max() - i.min())
    return (x)

df_norm = norm_func(wine)

# Storing the Total With SS of each cluster  and plotting elbow curve of TWSS
# No.of clusters = TWSS point on curve

TWSS = []
k = list(range(1,15 ))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")


# elbow curve descents at k=3

model = KMeans(n_clusters = 3)
model.fit(df_norm)
y_kmeans = model.predict(df_norm) # prediction
#each clusters number
model.labels_

#to know the centers (centroid) of each clusters w.r.t to each col in wine data
model.cluster_centers_
df_norm


plt.scatter(df_norm.iloc[:, 0], df_norm.iloc[:, 10], c=y_kmeans, s=50, cmap='viridis')
plt.legend(df_norm)
# Plotting a scatter plot for Type of Wine and its Alcohol content

centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);