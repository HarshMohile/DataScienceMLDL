import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


wine = pd.read_csv("D:\\360Assignments\\Submission\\10 PCA\\wine.csv")
wine.describe()

wine.head(178)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(wine)

scaled_data = scaler.transform(wine)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
scaled_data.shape

x_pca.shape
# from 14 features it  reduced its dimenson into 3 PC's

#Comparing first PCA with most variance with 2nd PCA with little lesser variance
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=wine['Type'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

#Comparing first PCA with most variacne with 3rd PCA with little lesser variance
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,2],c=wine['Type'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Third Principal Component')

fig, ax = plt.subplots()
ax.plot(x_pca[:,0], np.sin(x_pca[:,0]), '-b', label='First PCA')
ax.plot(x_pca[:,2], np.cos(x_pca[:,2]), '--r', label='Third PCA')
ax.axis('equal')
leg = ax.legend();


pca.components_[0]

pca.components_[1]

pca.components_[2]


df_comp = pd.DataFrame(pca.components_,columns=wine.columns)
# values are PCA values , columns are wine df columns
plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)
# Y -Axis is PCA Component and X-axis is the wine data with PCA values


#############hclust
# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
z = linkage(df_comp, method = "complete", metric = "euclidean")
z

plt.figure(figsize=(9, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()
df_comp

# Using AGNES with n-clusters:
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean").fit(df_comp) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)
df_comp['Belonging Cluster'] = cluster_labels
df_comp

df_comp.to_csv("Wine_compresse_data.csv", encoding = "utf-8")

import os
os.getcwd()

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

# Cumulative variance 
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1