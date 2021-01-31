import pandas as pd
import matplotlib as plt
import numpy as np
import seaborn as sns
df = pd.read_csv("D:\\360Assignments\\Submission\\15 Ensemble Techniques\\Assignment\\wbcd.csv")

##EDA
df.columns
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
## Mdoel Building


from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)
ada_clf.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, confusion_matrix

prediction_test =ada_clf.predict(X_test)
confusion_matrix(y_test, prediction_test)

'''
array([[76,  1],
       [ 3, 34]], dtype=int64)
'''
accuracy_score(y_test, prediction_test)
#0.9649122807017544

prediction_train =ada_clf.predict(X_train)
confusion_matrix(y_train, prediction_train)
'''
array([[280,   0],
       [  0, 175]], dtype=int64)
'''
accuracy_score(y_train, prediction_train)
#1.0

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(ada_clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print(np.mean(n_scores),np.std(n_scores))
#0.973611111111111 0.020191207400395834
n_scores

