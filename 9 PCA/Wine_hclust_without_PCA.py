import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

wine = pd.read_csv("D:\\360Assignments\\Submission\\10 PCA\\wine.csv")
wine.describe()

def normalize(i):
    x= (i-i.min())/(i.max()-i.min())
    return (x)

df_norm = normalize(wine)
df_norm.describe()
df_norm

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")
z
#complete linkage with euclidean for finding distance
# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Using AGNES with n-clusters:
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

wine['Belonging Cluster'] = cluster_labels

wine.head(178)

wine.groupby(['Type','Belonging Cluster']).groups
# grouping by cluster
grouped =wine.groupby('Belonging Cluster')

for name,group in grouped:
    print(name)
    print(group)

wine.groupby('Belonging Cluster').mean()

wine.to_csv("Wine_data.csv", encoding = "utf-8")

import os
os.getcwd()
