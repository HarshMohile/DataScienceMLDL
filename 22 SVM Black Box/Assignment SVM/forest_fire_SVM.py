# SVM 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

forest = pd.read_csv("D:\\360Assignments\\Submission\\22 SVM Black Box\\Assignment SVM\\forestfires.csv")
forest.describe()



forest.head()
forest.describe()

## isnull() comes from pd and  np,
forest.isnull().values.any()
forest.isnull().sum()

forest.columns

# remove the first 2 columns as dummy variables are already present
forest.drop(['month','day'],inplace=True,axis=1)

#EDA 
sns.pairplot(forest, hue="size_category", palette="Dark2")

sns.heatmap(forest.corr(), annot=True)

sns.kdeplot( forest['temp'], forest['area'],
                 cmap="plasma", shade=True, shade_lowest=False)


# Form a facetgrid using columns with a hue 
graph = sns.FacetGrid(forest, col ="rain",  hue ="size_category") 
# map the above form facetgrid with some attributes 
graph.map(plt.scatter, "temp", "monthnov", edgecolor ="w").add_legend()


from sklearn.model_selection import train_test_split
X = forest.drop('size_category',axis=1)
y = forest['size_category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train,y_train)

# model Evaulation 
predictions = svc_model.predict(X_test) 

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predictions))

'''
[[  1  46]
 [  0 109]]
'''

print(classification_report(y_test,predictions))

'''
           precision    recall  f1-score   support

       large       1.00      0.02      0.04        47
       small       0.70      1.00      0.83       109

    accuracy                           0.71       156
   macro avg       0.85      0.51      0.43       156
weighted avg       0.79      0.71      0.59       156
'''
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)


grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test,grid_predictions))

'''
[[ 36  11]
 [  6 103]]
'''
print(classification_report(y_test,grid_predictions))

'''
           precision    recall  f1-score   support

       large       0.86      0.77      0.81        47
       small       0.90      0.94      0.92       109

    accuracy                           0.89       156
   macro avg       0.88      0.86      0.87       156
weighted avg       0.89      0.89      0.89       156

Much better result after GridSearchCV 
'''

##--------------------------------- Now trying with different kernels  --------------------------
svc_lin_model = SVC(kernel="linear")
svc_lin_model.fit(X_train,y_train)

# model Evaulation 
predictions = svc_lin_model.predict(X_test) 

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predictions))

'''
[[ 45   2]
 [  0 109]]
'''

print(classification_report(y_test,predictions))

'''
          precision    recall  f1-score   support

       large       1.00      0.96      0.98        47
       small       0.98      1.00      0.99       109

    accuracy                           0.99       156
   macro avg       0.99      0.98      0.98       156
weighted avg       0.99      0.99      0.99       156

Linear perfomred better alone with kernel as Linear
'''