import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


diabetes = pd.read_csv("D:\\360Assignments\\Submission\\14 Decision Tree\\Assignment Dtree\\Diabetes.csv");

diabetes.columns

# Data Visualization
diabetes[[' Age (years)',' Class variable']].plot.box()

sns.pairplot(diabetes, hue=" Class variable", palette='viridis')

diabetes[' Age (years)'].plot.kde()
# Right  or positive skewness observed

# Checking NULL
diabetes[diabetes.isnull()].mean()


sns.heatmap(diabetes.corr(),cmap='coolwarm' , annot=True)
plt.title('diabetes.corr()')

diabetes[' Age (years)'].plot.hist(bins=20)
# Number of people getting Daibetes is after 20 and above but slowly descreases after reaching 40

sns.heatmap(diabetes.isnull(),yticklabels=False,cbar=False,cmap='viridis')


from sklearn.preprocessing import LabelEncoder
lb =LabelEncoder()
diabetes[' Class variable'] =lb.fit_transform(diabetes[" Class variable"])


diabetes.isnull().values.any()
diabetes.isnull().sum().sum()

np.any(np.isnan(diabetes))
np.all(np.isfinite(diabetes))

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

final_data = diabetes.apply(NormalizeData)


#################### MODEL SELECTION  TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split


X = final_data.drop(' Class variable',axis=1)
y = final_data[' Class variable']


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

         0.0       0.77      0.70      0.73       148
         1.0       0.54      0.63      0.58        83

    accuracy                           0.68       231
   macro avg       0.66      0.66      0.66       231
weighted avg       0.69      0.68      0.68       231
'''

print(confusion_matrix(y_test,predictions))
'''
[[104  44]
 [ 31  52]]
'''

## Random  Forest Model ##################################

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_pred))
'''

[[125  23]
 [ 33  50]]
'''
print(classification_report(y_test,rfc_pred))

'''

              precision    recall  f1-score   support

         0.0       0.79      0.84      0.82       148
         1.0       0.68      0.60      0.64        83

    accuracy                           0.76       231
   macro avg       0.74      0.72      0.73       231
weighted avg       0.75      0.76      0.75       231

Accuracy improved using Random forest from 68 % to 76%
'''
############ TREE VISUALIZATION
from IPython.display import Image  
from six import StringIO  
from sklearn.tree import export_graphviz
import pydot 
from sklearn import tree


#----------- TEXTUAL TREE
text_representation = tree.export_text(dtree)
print(text_representation)


fig, ax = plt.subplots(figsize=(12, 12))
tree.plot_tree(dtree.fit(X_train,y_train),max_depth=4, fontsize=8)
plt.show()