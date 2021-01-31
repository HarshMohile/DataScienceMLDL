import pandas as pd
import matplotlib as plt
import numpy as np
import seaborn as sns
df = pd.read_csv("D:\\360Assignments\\Submission\\15 Ensemble Techniques\\Assignment\\Diabetes_RF.csv")

##EDA
df.columns

df.isnull().values.any()
df.isnull().sum().sum()
#sns.pairplot(df, hue="diagnosis")

df.rename(columns={ " Class variable":"Daibetes_check" },inplace=True)

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Input and Output Split
X = df.loc[:, df.columns!="Daibetes_check"]
type(X)
y = df["Daibetes_check"]
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
array([[91, 16],
       [15, 32]], dtype=int64)
'''
accuracy_score(y_test, prediction_test)
#0.7987012987012987

prediction_train =ada_clf.predict(X_train)
confusion_matrix(y_train, prediction_train)
'''
array([[353,  40],
       [ 57, 164]], dtype=int64)
'''
accuracy_score(y_train, prediction_train)
#0.8420195439739414

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(ada_clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print(np.mean(n_scores),np.std(n_scores))
#0.7539872408293461 0.044374612219959045
n_scores



