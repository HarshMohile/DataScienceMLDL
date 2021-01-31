import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df= pd.read_csv("D:\\360Assignments\\Submission\\13 KNN\\glass.csv")
df

# type is our target class
'''
Standardize features by removing the mean and scaling to unit variance
'''

from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()

scaled_features = scaler.transform(df.drop('Type',axis=1))

# Columns are given since line before while std we applied it on .values only
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# Splitting the data in train test split
from sklearn.model_selection import train_test_split

# train_test_split(x, y)
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['Type'],
                                                    test_size=0.30)


#     Using KNN   Model fitting

from sklearn.neighbors import KNeighborsClassifier

# Startting with Overfitting by giving k=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)


#### Model Evaluation and Predictions

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))

'''
[[19  1  2  0  0  0]
 [ 4 14  1  1  1  0]
 [ 6  2  1  0  0  0]
 [ 0  1  0  4  0  0]
 [ 0  1  0  0  1  0]
 [ 0  1  0  0  0  5]]
'''
print(classification_report(y_test,pred))
'''
                precision    recall  f1-score   support

           1       0.66      0.86      0.75        22
           2       0.70      0.67      0.68        21
           3       0.25      0.11      0.15         9
           5       0.80      0.80      0.80         5
           6       0.50      0.50      0.50         2
           7       1.00      0.83      0.91         6

    accuracy                           0.68        65
   macro avg       0.65      0.63      0.63        65
weighted avg       0.65      0.68      0.65        65
'''
crtab= pd.crosstab(y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 
#reports outcome is pretty low as it should be since k value is not appropriate for our dataset.

#### EVALAUATING K VALUE

error_rate = []


for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# In this error rate plot we  get curve or descent somewhee around k=3 .
# Here Accuracy is even worse but atleast its not overfitting , problem arises when number of classes are more.

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=3')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

