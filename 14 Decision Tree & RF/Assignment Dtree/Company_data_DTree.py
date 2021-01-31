import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

company =pd.read_csv("D:\\360Assignments\\Submission\\14 Decision Tree\\Assignment Dtree\\Company_Data.csv")

# EDA 
sns.pairplot(company,hue='ShelveLoc',palette='Set1')

company[company.isnull()].mean()

sns.heatmap(company.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sns.set_style('whitegrid')
sns.countplot(x='Urban',hue='ShelveLoc',data=company,palette='RdBu_r')
#Urban  with ShelvLoc seems to have higherfreq appearance in dataset compared to Non-Urban   


# ShelevLoc quality Medium is of higher freq disrtr ver the entire dataset.

## Categorical data for ''Sales' feature

#Sales_category =pd.cut(company.Sales,bins=[],labels=[1,2,3,4],precision=3)

#company.insert(0,'Sales_category',Sales_category)

#company = company[~company['Sales_category'].isnull()]
#company[['Sales_category']] = company[['Sales_category']].astype(int)

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
company["Sales"] = lb.fit_transform(company["Sales"])
company['ShelveLoc'] =lb.fit_transform(company["ShelveLoc"])
company['Urban'] =lb.fit_transform(company["Urban"])
company['US'] =lb.fit_transform(company["US"])

Sales_category =pd.cut(company['Sales'],bins=[0,100,400],labels=[1,2],precision=3)
company.insert(0,'Sales_category',Sales_category)

# data type is category and we convert into int for our model

company = company[~company['Sales_category'].isnull()]
company[['Sales_category']] = company[['Sales_category']].astype(int)

# below Average Sales have higherst value counts

#  Advertising is an categorical feature for X_train that cannot be converted into string so we use dummy variable
#ShelveLoc_feats = ['ShelveLoc']
#final_data = pd.get_dummies(comp,columns=ShelveLoc_feats,drop_first=True)
'''
dummies0 = pd.get_dummies(company['ShelveLoc']).rename(columns=lambda x: 'ShelveLoc_' + str(x))
final_data = pd.concat([company,dummies0],axis=1)

#Sales_feats = ['Sales_category']
#final_data = pd.get_dummies(comp,columns=Sales_feats,drop_first=True)


## Urban and US has Yes ,No category so it has to be convrted into 1,0
dummies = pd.get_dummies(final_data['Urban']).rename(columns=lambda x: 'Urban_' + str(x))
final_data = pd.concat([final_data,dummies],axis=1)


dummies1 = pd.get_dummies(final_data['US']).rename(columns=lambda x: 'US_' + str(x))
final_data = pd.concat([final_data,dummies1],axis=1)



final_data = final_data.drop(['Urban', 'US'],axis=1)
final_data = final_data.drop(['ShelveLoc'],axis=1)
'''
company.isnull().values.any()
company.isnull().sum().sum()

np.any(np.isnan(company))
np.all(np.isfinite(company))

company.dropna()
#final_data = final_data.reset_index()

company.info()


company =company.drop('Sales',axis=1)
# Normalize the data
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

final_data = company.iloc[1:].apply(NormalizeData)

final_data.info()

#################### MODEL SELECTION  TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
#X = final_data1.drop('Sales_category',axis=1)
#y = final_data1['Sales_category']

X = final_data.drop('Sales_category',axis=1)
y = final_data['Sales_category']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

## importing Decision Tree classifier 
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)




from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))

'''
             precision    recall  f1-score   support

         0.0       0.65      0.57      0.61        35
         1.0       0.83      0.87      0.85        85

    accuracy                           0.78       120
   macro avg       0.74      0.72      0.73       120
weighted avg       0.78      0.78      0.78       120
'''
print(confusion_matrix(y_test,predictions))
'''
[[20 15]
 [11 74]]
'''

############ TREE VISUALIZATION

from IPython.display import Image  
from six import StringIO  
from sklearn.tree import export_graphviz
import pydot 
from sklearn import tree
'''
pip install six
pip install pydot
'''
#----------- TEXTUAL TREE
text_representation = tree.export_text(dtree)
print(text_representation)




features = list(final_data.columns)
features

fig, ax = plt.subplots(figsize=(12, 12))
tree.plot_tree(dtree.fit(X_train,y_train),max_depth=4, fontsize=8)
plt.show()

## Random  Forest Model ##################################

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_pred))
'''
[[16 21]
 [ 5 78]]
'''
print(classification_report(y_test,rfc_pred))
'''
              precision    recall  f1-score   support

         0.0       0.76      0.43      0.55        37
         1.0       0.79      0.94      0.86        83

    accuracy                           0.78       120
   macro avg       0.77      0.69      0.70       120
weighted avg       0.78      0.78      0.76       120

Random Forest gave better F1(Accuracy) for Class 0 and Class 1 than Dtree 
'''