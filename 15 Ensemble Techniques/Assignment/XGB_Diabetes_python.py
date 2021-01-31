import pandas as pd
import matplotlib as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("D:\\360Assignments\\Submission\\15 Ensemble Techniques\\Assignment\\Diabetes_RF.csv")

df.head()

df.describe()
df.columns

df[' Class variable'].value_counts()

df.rename(columns={' Class variable':'Diabetes_check'},inplace=True)
'''
NO     500
YES    268
Name:  Class variable, dtype: int64
'''

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

df["Diabetes_check"] = lb.fit_transform(df["Diabetes_check"])


# Input and Output Split
X = df.loc[:, df.columns!="Diabetes_check"]
type(X)
y = df["Diabetes_check"]
type(y)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)


xgb_clf.fit(x_train, y_train)    


from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data

prediction_test= xgb_clf.predict(x_test)

confusion_matrix(y_test, prediction_test)

'''
array([[84, 23],
       [12, 35]], dtype=int64)
'''
accuracy_score(y_test, prediction_test)
#0.7727272727272727

# Evaluation on Train Data

prediction_train= xgb_clf.predict(x_train)

confusion_matrix(y_train, prediction_train)

'''
array([[393,   0],
       [  0, 221]], dtype=int64)
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
#0.8116883116883117

print(confusion_matrix( y_test, pred_cv_test))

'''
[[96 11]
 [18 29]]
'''
grid_search.best_params_



'''
{'colsample_bytree': 0.6,
 'gamma': 5,
 'max_depth': 3,
 'min_child_weight': 1,
 'subsample': 1.0}
'''

