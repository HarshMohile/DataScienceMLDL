# SVM 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

sal_test = pd.read_csv("D:\\360Assignments\\Submission\\22 SVM Black Box\\Assignment SVM\\SalaryData_Test.csv")
sal_test.describe()


sal_train = pd.read_csv("D:\\360Assignments\\Submission\\22 SVM Black Box\\Assignment SVM\\SalaryData_Train.csv")
sal_train


# EDA  on train data
sal_train.info

sns.heatmap(sal_train.isna())
# no Na values 
sns.heatmap(sal_test.isna())

sal_train.value_counts()
sal_train.workclass.unique()

df_merged = pd.concat([sal_test,sal_train],axis=1)





## isnull() comes from pd and  np,
sal_train.isnull().values.any()
sal_train.isnull().sum()
sal_test.isnull().values.any()
sal_test.isnull().sum()


# Merging 2 dataframe 
df_merged.isnull().values.any()
df_merged.isnull().sum()


df_merged.dropna()


# converting categorical in X into numeric for both X_Train and Y_train

# Looking for categorical variables in X
categorical_train = [var for var in sal_train.columns if sal_train[var].dtype=='O']

categorical_test = [var for var in sal_test.columns if sal_test[var].dtype=='O']


['workclass',
 'education',
 'maritalstatus',
 'occupation',
 'relationship',
 'race',
 'sex',
 'native',
 'Salary']

sal_train.education.value_counts().count()



sal_train1 =pd.get_dummies(sal_train, columns =["workclass","education"
                                               ,"maritalstatus","occupation"
                                               ,"relationship","sex","race"
                                               ,"native"],drop_first=True)


sal_test1 =pd.get_dummies(sal_test, columns =["workclass","education"
                                               ,"maritalstatus","occupation"
                                               ,"relationship","sex","race"
                                               ,"native"],drop_first=True)


#sal_train1.drop([["workclass","education","maritalstatus","occupation","relationship","sex","native"]],axis=1,inplace=True)




# Splitting the data 
X_train = sal_train1.drop("Salary",axis=1)
y_train = sal_train1["Salary"]

X_test  = sal_test1.drop("Salary",axis=1)
y_test  = sal_test1["Salary"]




from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train,y_train)

# model Evaulation 
predictions = svc_model.predict(X_test) 

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predictions))

'''
[[10997   363]
 [ 2703   997]]
'''

print(classification_report(y_test,predictions))

'''              precision    recall  f1-score   support

       <=50K       0.80      0.97      0.88     11360
        >50K       0.73      0.27      0.39      3700

    accuracy                           0.80     15060
   macro avg       0.77      0.62      0.64     15060
weighted avg       0.79      0.80      0.76     15060
'''

##--------------------------------- Now trying with different kernels  --------------------------
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(X_train, y_train)
pred_test_rbf = model_rbf.predict(X_test)

np.mean(pred_test_rbf==y_test)
# 0.7964143426294821



