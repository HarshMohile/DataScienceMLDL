import pandas as pd
import matplotlib.pylab as plt

airline= pd.read_excel("D:\\360Assignments\\Submission\\DataMining using Hclust ans Kmeans 4 5\\EastWestAirlines.xlsx")

airline.describe()
airline_data = airline.drop(["ID#"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(airline_data)
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(16, 9));plt.title('Hierarchical Clustering Dendrogram for Airline');
plt.xlabel('Index');
plt.ylabel('Distance')

sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()



