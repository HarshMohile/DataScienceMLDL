import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df= pd.read_csv("D:\\360Assignments\\Submission\\13 KNN\\Zoo.csv")
df
#removing Animal name column and Type is our Target Class
zoo1 = df.iloc[:, 1:18]



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop('Type',axis=1))


scaled_features = scaler.transform(df.drop('Type',axis=1))

# Columns are given since line before while std we applied it on .values only
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# Splitting the data in train test split
from sklearn.model_selection import train_test_split

# train_test_split(x, y)
X_train, X_test, y_train, y_test = train_test_split(df_feat,zoo1['type'],
                                                    test_size=0.30)

#     Using KNN   Model fitting

from sklearn.neighbors import KNeighborsClassifier

# Startting with Overfitting by giving k=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))


print(classification_report(y_test,pred))
'''
              precision    recall  f1-score   support

           1       1.00      1.00      1.00         8
           2       1.00      1.00      1.00        10
           3       0.00      0.00      0.00         0
           4       1.00      1.00      1.00         4
           5       1.00      0.50      0.67         2
           6       0.80      1.00      0.89         4
           7       1.00      0.67      0.80         3

    accuracy                           0.94        31
   macro avg       0.83      0.74      0.77        31
weighted avg       0.97      0.94      0.94        31
'''

# Overfitting the dataset caused  problems by giving k =1
crtab= pd.crosstab(y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 
'''
Predictions  1   2  3  4  5  6  7
Actual                           
1            8   0  0  0  0  0  0
2            0  10  0  0  0  0  0
4            0   0  0  4  0  0  0
5            0   0  1  0  1  0  0
6            0   0  0  0  0  4  0
7            0   0  0  0  0  1  2
'''

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


knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=5')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')


'''
[[16  0  0  0  0  0]
 [ 0  3  0  0  0  0]
 [ 0  1  1  0  0  0]
 [ 0  0  0  5  0  0]
 [ 0  0  0  0  1  0]
 [ 0  0  0  0  2  2]]
'''
print(classification_report(y_test,pred))

'''
         precision    recall  f1-score   support

           1       1.00      1.00      1.00        16
           2       0.75      1.00      0.86         3
           3       1.00      0.50      0.67         2
           4       1.00      1.00      1.00         5
           6       0.33      1.00      0.50         1
           7       1.00      0.50      0.67         4

    accuracy                           0.90        31
   macro avg       0.85      0.83      0.78        31
weighted avg       0.95      0.90      0.91        31
'''

# 