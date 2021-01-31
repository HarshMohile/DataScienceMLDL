import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

wine = pd.read_csv("D:\\360Assignments\\Submission\\10 PCA\\wine.csv")
wine.describe()
wine.head()  

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(wine)
scaled_data = scaler.transform(wine)


scaled_data

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(scaled_data)

scaled_data 

x_pca = pca.transform(scaled_data)
x_pca  

scaled_data.shape

x_pca.shape
#Reduced from 14 to 3 Components  
#Comparing first PCA with most variance with 2nd PCA with little lesser variance
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=wine['Type'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

pca.components_[0]
pca.components_[1]
pca.components_[2]

df_comp = pd.DataFrame(pca.components_,columns=wine.columns)
# values are PCA values , columns are wine df columns
plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)
# Y -Axis is PCA Component and X-axis is the wine data with PCA values


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


#Finding the TWSS  of each cluster
TWSS = []
k = list(range(1,4))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_comp)
    TWSS.append(kmeans.inertia_)
    
TWSS

plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")


model = KMeans(n_clusters = 3)
model.fit(df_comp)
y_kmeans = model.predict(df_comp) # prediction
#each clusters number
model.labels_

#to know the centers (centroid) of each clusters w.r.t to each col in wine data
model.cluster_centers_

df_comp

plt.scatter(df_comp.iloc[:, 0], df_comp.iloc[:, 1], c=y_kmeans, s=50, cmap='viridis')
# Plotting a scatter plot for Type of Wine and its Alcohol content

centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

