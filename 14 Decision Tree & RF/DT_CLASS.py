import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 

data = pd.read_csv("D:\\Train\\DT\\credit.csv")
data.isnull()
data.isnull().sum()
data.dropna()
data.columns
data = data.drop(["phone"], axis = 1)

#converting into binary
lb = LabelEncoder()
data["checking_balance"] = lb.fit_transform(data["checking_balance"])
data["credit_history"] = lb.fit_transform(data["credit_history"])
data["purpose"] = lb.fit_transform(data["purpose"])
data["savings_balance"] = lb.fit_transform(data["savings_balance"])
data["employment_duration"] = lb.fit_transform(data["employment_duration"])
data["other_credit"] = lb.fit_transform(data["other_credit"])
data["housing"] = lb.fit_transform(data["housing"])
data["job"] = lb.fit_transform(data["job"])

#data["default"]=lb.fit_transform(data["default"])

data['default'].unique()
data['default'].value_counts()
colnames = list(data.columns)

predictors = colnames[:15]
target = colnames[15]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

'''
# As we can say that its overfitting model and Drawback of Decision tree is that it often gives overfitting models.
#so to improve we have to use the ensemble algorithm to overcome overfitting problem.

___________________________________________________________________
________________________________________________________________________________
______________________________________________________________________________
_______________________________________________________________________________
________________'''

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 19:45:30 2020

@author: hp
"""
import numpy as np
import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# In[87]:


X = data.values[:, :15]
Y = data.values[:,15]
#xyz=data.values[180:181,:13]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# In[88]:


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
from sklearn import tree
clf = tree.DecisionTreeClassifier()

#Once trained, you can plot the tree with the plot_tree function:
%matplotlib inline
tree.plot_tree(clf_gini) 

from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(125,200))
_ = tree.plot_tree(clf_gini,filled=True)


# In[92]:


y_pred = clf_gini.predict(X_test)
#y_pred
print ("\nAcuraccy score ::: ",accuracy_score(y_test,y_pred)*100)




