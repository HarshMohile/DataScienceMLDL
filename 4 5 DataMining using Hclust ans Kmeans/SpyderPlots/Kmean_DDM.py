import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# Generating random uniform numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=["X","Y"])
df_xy
df_xy.X = X
df_xy.Y = Y
df_xy

df_xy.plot(x="X", y="Y", kind = "scatter")

model1 = KMeans(n_clusters = 4).fit(df_xy)

df_xy.plot(x = "X", y = "Y", c = model1.labels_, kind="scatter", s = 10, cmap = plt.cm.coolwarm)

# Kmeans on University Data set 
Univ1 = pd.read_excel("D:\\Train\\K-mean\\University_Clustering.xlsx")
#sdsf= pd.read_excel("D:/Train/K-mean/University_Clustering.xlsx")
#pd.read_excel('D:\\Train\\K-mean\\University_Clustering.xlsx')
Univ1 = Univ1.iloc[:,2:8]
Univ1.boxplot(vert=0)
Univ1.boxplot(Univ1['Accept'])

from scipy import stats

#Univ1[(np.abs(stats.zscore(Univ1)) < 3).all(axis=1)]

'''Z	=	standard score
x	=	observed value
mu	=	mean of the sample
sigma	=	standard deviation of the sample
'''

'''code means that:

For each column, first, it computes the Z-score of each value in the column,
 relative to the column mean and standard deviation.

Then it takes the absolute of Z-score because the direction does not matter,
 only if it is below the threshold.

all(axis=1) ensures that for each row, all columns satisfy the constraint.

Finally, the result of this condition is used to index the dataframe.'''

Univ1.head #printout the records
Univ1.shape #dim
Univ1.boxplot()
Univ1.info()#structure
Univ1.isnull().sum()
Univ1.dropna()
plt.scatter(Univ1)
X= Univ1['SAT']
Y= Univ1['Accept']
plt.scatter(Univ1['SAT'], Univ1['Accept'])
plt.scatter(X,Y)



'''Boxplots are a standardized way of displaying the distribution of data based 
on a five number summary (“minimum”, first quartile (Q1), median, third quartile (Q3),
                          and “maximum”).
median (Q2/50th Percentile): the middle value of the dataset.
first quartile (Q1/25th Percentile): the middle number between the smallest number 
(not the “minimum”) and the median of the dataset.
third quartile (Q3/75th Percentile): the middle value between the median and the 
highest value (not the “maximum”) of the dataset.
interquartile range (IQR): 25th to the 75th percentile.
whiskers (shown in blue)
outliers (shown as green circles)
“maximum”: Q3 + 1.5*IQR
“minimum”: Q1 -1.5*IQR
'''
'''Numerical vs. Numerical
1. Scatterplot
2. Heatmap for correlation


Categorical vs. Numerical
1. Bar chart



Two Categorical Variables
1. Bar chart

'''

Univ1.corr()

Univ1.describe()

Univ = pd.read_excel("D:\\Train\\K-mean\\University_Clustering.xlsx")
Univ = Univ.drop(["State"], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())/(i.max() - i.min())
    return (x)



# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Univ.iloc[:, 1:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');
plt.xlabel("No_of_Clusters");
plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 5)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
Univ['clust'] = mb # creating a  new column and assigning it to new column 

Univ.head()
df_norm.head()

Univ = Univ.iloc[:,[7,0,1,2,3,4,5,6]]
Univ.head()

Univ.iloc[:, 2:8].groupby(Univ.clust).mean()

Univ.to_csv("Kmeans_university.csv", encoding = "utf-8")

import os
os.getcwd()
