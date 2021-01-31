import pandas as pd
import numpy as np
import matplotlib.pylplot as plt

from sklearn.cluster import	KMeans
from sklearn.datasets import make_blobs

airline= pd.read_excel("D:\\360Assignments\\Submission\\DataMining using Hclust ans Kmeans 4 5\\EastWestAirlines.xlsx")

#remove the nominal data like ID
airline.describe()
air_data = airline.drop(["ID#"], axis = 1)

#To normalize the data 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

df_norm = norm_func(air_data)


#elbow curve to find the best number of clusters
TWSS = []
k = list(range(1, 12))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# elbow curve descents at k=2

model = KMeans(n_clusters = 2)
model.fit(df_norm)
y_kmeans = model.predict(df_norm) # prediction
#each clusters number
model.labels_

#to know the center s of each clusters
model.cluster_centers_

plt.scatter(df_norm.iloc[:, 0], df_norm.iloc[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


