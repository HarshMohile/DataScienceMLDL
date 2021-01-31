import pandas as pd
import matplotlib as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("D:\\360Assignments\\Submission\\15 Ensemble Techniques\\Assignment\\wbcd.csv")

df.head()

df.describe()

df['diagnosis'].value_counts

# n-1 dummy variables will be created for n categories
#df1 = pd.get_dummies(df, columns = ["diagnosis"],drop_first=True)


#Label Enciding because target variable needs to be integers 1,2,3..
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

df["diagnosis"] = lb.fit_transform(df["diagnosis"])

#remove Id column
df.drop('id',inplace=True,axis=1)

df.isnull().values.any()
df.isnull().sum().sum()
#sns.pairplot(df, hue="diagnosis")
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Input and Output Split
X = df.loc[:, df.columns!="diagnosis"]
type(X)
y = df["diagnosis"]
type(y)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)

# n_jobs – Number of parallel threads used to run xgboost.
# learning_rate (float) – Boosting learning rate (xgb’s “eta”)

xgb_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data

prediction_test= xgb_clf.predict(x_test)

confusion_matrix(y_test, prediction_test)

'''
array([[77,  0],
       [ 3, 34]], dtype=int64)
'''
accuracy_score(y_test, prediction_test)
#0.9736842105263158

# Evaluation on Train Data

prediction_train= xgb_clf.predict(x_train)

confusion_matrix(y_train, prediction_train)

'''
array([[280,   0],
       [  0, 175]], dtype=int64)
'''
accuracy_score(y_train, prediction_train)
#1.0


xgb.plot_importance(xgb_clf)

### Model
xgb_clf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42,
                            objective='binary:logistic')

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

# Grid Search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(xgb_clf, params, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(x_train, y_train)

cv_xg_clf = grid_search.best_estimator_

# Evaluation on Testing Data with model with hyperparameter

pred_cv_test = cv_xg_clf.predict(x_test)

accuracy_score(y_test,pred_cv_test )
#0.956140350877193

print(confusion_matrix( y_test, pred_cv_test))

'''
[[76  1]
 [ 4 33]]
'''
grid_search.best_params_



'''
{'colsample_bytree': 0.6,
 'gamma': 1.5,
 'max_depth': 5,
 'min_child_weight': 1,
 'subsample': 1.0}
'''

