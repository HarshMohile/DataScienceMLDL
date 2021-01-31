import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt
import seaborn as sns


G_fb = pd.read_csv("D:\\360Assignments\\Submission\\8 Network Analytics\\facebook.csv")
G_ln = pd.read_csv("D:\\360Assignments\\Submission\\8 Network Analytics\\linkedin.csv")
G_insta = pd.read_csv("D:\\360Assignments\\Submission\\8 Network Analytics\\Instagram.csv")

graph = nx.from_numpy_matrix(G_fb.values)
print(nx.info(graph))
#Name:  for Facebook
#Type: Graph
#Number of nodes: 9
#Number of edges: 9
#Average degree:   2.0000