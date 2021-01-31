
import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
elec = pd.read_csv("D:\\360Assignments\\Submission\\20 Logistic Regression\\Assignment Log Reg\\election_data.csv", sep = ",")



## isnull() comes from pd and  np,
elec.isnull().values.any()
elec.isnull().sum()
elec.dropna()

####EDA 

plt.bar(height = elec['Amount Spent'], x= elec['Popularity Rank'])
sns.heatmap(elec.corr(),cmap="coolwarm",annot=True)

# Form a facetgrid using columns with a hue 
graph = sns.FacetGrid(elec, col ="Popularity Rank",  hue ="Result") 
# map the above form facetgrid with some attributes 
graph.map(plt.scatter, "Election-id", "Year", edgecolor ="w").add_legend() 

sns.pairplot(elec,hue='Result',palette='bwr')


#############
elec.columns
elec.rename(columns={"Amount Spent":"AmtSpent"},inplace =True)
elec.rename(columns={"Popularity Rank":"PopRank"},inplace =True)
elec.rename(columns={"Election-id":"ElecId"},inplace =True)
#----------------------------------------------------------------------------------------------------
# Model building 
# import statsmodels.formula.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#logit_model = sm.logit('Result ~ ElecId + AmtSpent + PopRank + Year', data = elec).fit()
# sm logit didnt work since the dataset has quasi perfect 0 and 1 (equal 0 and 1 )
###------------------------ Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
# X and y  variables
X= elec.drop('Result',axis=1)
y= elec['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

logmodel = LogisticRegression(penalty='l2')  # only supports l2 

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

#    Metrices
fpr, tpr, thresholds = roc_curve(y_test, predictions)

# Optimal Thresholds
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]   # thresholds[59]  index location
optimal_threshold
                    #2
                    
                    
import pylab as pl

# Create a Dataframe with these columns names  for ROC: true positive rate (tpr)  false positive rate (fpr)****************************************
'''
    fpr    tpr   1-fpr   tf i.e (tpr - (1-fpr))   thresholds  
'''

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),
                    'tpr' : pd.Series(tpr, index = i),
                    '1-fpr' : pd.Series(1-fpr, index = i), 
                    'tf' : pd.Series(tpr - (1-fpr), index = i),
                    'thresholds' : pd.Series(thresholds, index = i)})

roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')  # above all for plot
ax.set_xticklabels([]) # prints in  console

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)    # 0.50000


