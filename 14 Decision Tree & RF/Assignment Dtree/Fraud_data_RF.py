import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fraud = pd.read_csv("D:\\360Assignments\\Submission\\14 Decision Tree\\Assignment Dtree\\Fraud_check.csv");

fraud.columns

from sklearn.preprocessing import LabelEncoder 
lb= LabelEncoder()
fraud["Undergrad"] = lb.fit_transform(fraud["Undergrad"])
fraud["Marital.Status"] = lb.fit_transform(fraud["Marital.Status"])
fraud["Urban"] = lb.fit_transform(fraud["Urban"])

# Risky and Good 
fraud['Taxable'] = fraud['Taxable.Income'].apply( lambda x: "Risky" if  x<=30000  else "Good")

final_data = fraud.drop('Taxable.Income', axis=1)

final_data["Taxable"] = lb.fit_transform(final_data["Taxable"])

# checking for NA or NAn for Null
final_data.isnull().values.any()
final_data.isnull().sum().sum()

np.any(np.isnan(final_data))
np.all(np.isfinite(final_data))

final_data.dropna()
#final_data = final_data.reset_index()

final_data.info()

# Normalize for the opulation data column
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


final_data = final_data.apply(NormalizeData)

final_data.info()


#################### MODEL SELECTION  TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
#X = final_data1.drop('Sales_category',axis=1)
#y = final_data1['Sales_category']

X = final_data.drop('Taxable',axis=1)
y = final_data['Taxable']


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

         0.0       0.78      0.70      0.74       141
         1.0       0.21      0.28      0.24        39

    accuracy                           0.61       180
   macro avg       0.49      0.49      0.49       180
weighted avg       0.66      0.61      0.63       180
'''
print(confusion_matrix(y_test,predictions))
'''
[[99 42]
 [28 11]]
'''

## Random  Forest Model ##################################

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=500)
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

         0.0       0.80      0.92      0.85       144
         1.0       0.14      0.06      0.08        36

    accuracy                           0.74       180
   macro avg       0.47      0.49      0.47       180
weighted avg       0.66      0.74      0.70       180
'''
# Accuracy improved from 61 to 74 by using Dtree to Random Forest
