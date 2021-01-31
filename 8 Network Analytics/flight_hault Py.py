import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt
import seaborn as sns


G = pd.read_csv("D:\\360Assignments\\Submission\\8 Network Analytics\\flight_hault.csv",header=None)
G.columns= ["ID","Name","City","Country","IATA_FAA","ICAO","Latitude","Longitude","Altitude","Time","DST","Tz database time"]

conn = pd.read_csv("D:\\360Assignments\\Submission\\8 Network Analytics\\connecting_routes.csv",header=None)

#Data Cleaning
sns.heatmap(conn.isnull(),cmap="viridis")
#conn.iloc[:,:-2] will drop last 2 column

conn1 = conn.dropna()
conn1 = conn.drop(6,axis=1)
conn1.columns= ["flights", " ID", "main Airport","main Airport ID", "Destination ", "Destination ID", "haults", "machinary"]


g = nx.Graph()

g = nx.from_pandas_edgelist(conn1, source = 'main Airport', target = 'Destination ')
print(nx.info(g))
#Type: Graph
#Number of nodes: 3425
#Number of edges: 19257
#Average degree:  11.2450

b = nx.degree_centrality(g)  # Degree Centrality
print(max(b)) 
 #ZYl hs the highest degree of centrality i.e Al nodes either connect other nodes via ZYL airport
 
pos = nx.spring_layout(g, k = 0.15)
nx.draw_networkx(g, pos, node_size = 25, node_color = 'blue')



# closeness centrality
closeness = nx.closeness_centrality(g)
print(closeness)
print(max(closeness))
#ZYL

conn1[conn1['main Airport']=='ZYL'].value_counts()
#flights   ID    main Airport  main Airport ID  Destination   Destination ID  haults  machinary
#VQ       11948  ZYL           3074             DAC           3076            0       ER4          1
#RX       19676  ZYL           3074             DAC           3076            0       DH8          1
#BG       1359   ZYL           3074             DAC           3076            0       313 772      1
#4H       8463   ZYL           3074             DAC           3076            0       313          1

 # ZYL all destinations were DAC airport .ie  its outdegree
 
conn1[conn1['main Airport']=='ZYL'].count()  # 4

## Betweeness Centrality 
b = nx.betweenness_centrality(g) # Betweeness_Centrality
print(max(b))
#ZYL max betweeness centrality

## Eigen-Vector Centrality
evg = nx.eigenvector_centrality(g) # Eigen vector centrality
print(max(evg))
#ZYL has maximum influence.

# cluster coefficient
cluster_coeff = nx.clustering(g)
cluster_zyl =nx.clustering(g,'ZYL') #0.8333333333333334
# If cluster coefficient is closer to 1 it is clique.
print(cluster_coeff)

#ZYL
#clustering coefficient is a measure of the degree to which nodes in a graph tend to cluster together
# 2 nodes that are connected are likely to be apart of major network
# Average clustering
cc = nx.average_clustering(g) 
print(cc)
#0.4870933566129556

