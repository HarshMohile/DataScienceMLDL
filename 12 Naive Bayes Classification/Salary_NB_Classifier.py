import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

salary_train= pd.read_csv("D:\\360Assignments\\Submission\\12 Naive Bayes Classification\\SalaryData_Train.csv")
salary_test = pd.read_csv("D:\\360Assignments\\Submission\\12 Naive Bayes Classification\\SalaryData_Test.csv")


############# EDA ####################
salary_test.columns
salary_train.columns

salary_train.info

sns.heatmap(salary_train.isna())
# no Na values 
sns.heatmap(salary_test.isna())

salary_train.value_counts()
salary_train.workclass.unique()

#salary_train[salary_train['Salary']=='<=50']['workclass']


categorical = [var for var in salary_train.columns if salary_train[var].dtype=='O']

salary_train[categorical]

numerical = [var for var in salary_train.columns if salary_train[var].dtype!='O']


salary_train[numerical]


X =salary_train.drop(['Salary'],axis=1)
y =salary_train['Salary']

############# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


X_train.shape, X_test.shape

X_train.isnull().mean()

'''['workclass',
 'education',
 'maritalstatus',
 'occupation',
 'relationship',
 'race',
 'sex',
 'native'] ARE TO BE normalized by creating dummy variables  or using One-Hot Encoders'''


#dummyworkclass =pd.get_dummies(X_train['workclass'])
#dummyworkclass.head()

import category_encoders as ce


encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 
                                 'race', 'sex', 'native'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

X_train.head()
X_test


cols = X_train.columns
# Nomralize the data since age will highly incluence the data to move in predictions
#from sklearn.preprocessing import RobustScaler

#scaler = RobustScaler()

#X_train = scaler.fit_transform(X_train)

#X_test = scaler.transform(X_test)
  
type(X_test)
type(X_train)
#########################  Scaling data from 0,1 since Multnomial only accepts (0,1) and not negative values
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

X_train_norm = X_train.apply(NormalizeData)

X_test_norm = X_test.apply(NormalizeData)

#X_train =pd.DataFrame(X_train, columns=[cols])
#X_test =pd.DataFrame(X_test, columns=[cols])
#X_train.head()


#####################  MODEL TRAINING #################
from sklearn.naive_bayes import MultinomialNB

mb= MultinomialNB()

mb.fit(X_train_norm,y_train)

#################### PREDICT THE MODEL ################
 
y_pred = mb.predict(X_test_norm)
y_pred


#################  MODEL EVALUATION ###################

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,y_pred))


'''             precision    recall  f1-score   support

       <=50K       0.90      0.80      0.85      6798
        >50K       0.56      0.74      0.64      2251

    accuracy                           0.79      9049
   macro avg       0.73      0.77      0.74      9049
weighted avg       0.82      0.79      0.80      9
'''

print(confusion_matrix(y_test,y_pred))
'''
[[5460 1338]
 [ 577 1674]]
'''

pd.crosstab(y_pred, y_test)   

'''Salary   <=50K   >50K
row_0                
 <=50K    5460    577
 >50K     1338   1674 
'''