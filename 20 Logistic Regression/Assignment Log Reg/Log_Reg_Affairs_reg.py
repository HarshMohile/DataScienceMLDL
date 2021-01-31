
import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
aff = pd.read_csv("D:\\360Assignments\\Submission\\20 Logistic Regression\\Assignment Log Reg\\Affairs.csv", sep = ",")

aff.columns
'''
Index(['naffairs', 'kids', 'vryunhap', 'unhap', 'avgmarr', 'hapavg', 'vryhap',
       'antirel', 'notrel', 'slghtrel', 'smerel', 'vryrel', 'yrsmarr1',
       'yrsmarr2', 'yrsmarr3', 'yrsmarr4', 'yrsmarr5', 'yrsmarr6'],
      dtype='object')
'''
aff= aff.drop("Unnamed: 0",axis=1)

aff.head()

aff.describe()

## isnull() comes from pd and  np,
aff.isnull().values.any()
aff.isnull().sum()
aff.dropna()

'''
# Converting N affairs to 0,1 by LabelEncoder
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
aff["naffairs"] = lb.fit_transform(aff["naffairs"]) 
'''
# converting (0,7,13, ) into 0 or 1 (either there was affair nor not)
aff['naffairs']=aff['naffairs'].apply(lambda x :1  if x>0 else 0)

###*************************************** EDA 
sns.pairplot(aff,hue='kids',palette='bwr')



#----------------------------------------------------------------------------------------------------
# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('naffairs ~ kids + vryunhap + unhap + avgmarr + hapavg' +
                          ' + vryhap + antirel  +notrel + slghtrel +  smerel + vryrel'+
                          '+ yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5 + yrsmarr6', data = aff).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

# X and y  variables
X= aff.drop('naffairs',axis=1)
y= aff['naffairs']

pred = logit_model.predict(X)   # predict(X_test)

#    Metrices
fpr, tpr, thresholds = roc_curve(y, pred)

# Optimal Thresholds
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]   # thresholds[59]  index location
optimal_threshold
                    #0.2521571570135329
                    
                    
import pylab as pl

# Create a Dataframe with these columns names  for ROC: true positive rate (tpr)  false positive rate (fpr)****************************************
'''
    fpr    tpr   1-fpr   tf i.e (tpr - (1-fpr))   thresholds  
'''

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),
                    'tpr' : pd.Series(tpr, index = i),
                    '1-fpr' : pd.Series(1-fpr, index = i), 
                    'tf' : pd.Series(tpr - (1-fpr), index = i),
                    'thresholds' : pd.Series(thresholds, index = i)})

roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')  # above all for plot
ax.set_xticklabels([]) # prints in  console

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)    # 0.720880

# filling all the cells with zeroes
aff["pred"] = np.zeros(601)

# taking threshold value and above the prob value will be treated as correct value 
aff.loc[pred > optimal_threshold, "pred"] = 1

# classification report
classification = classification_report(aff["pred"], aff["naffairs"])
classification

'''
 '              precision    recall  f1-score   support\n\n         
         0.0       0.71      0.86      0.77       370\n         
         1.0       0.65      0.42      0.51       231\n\n    
         accuracy                                   0.69       
         601\n   macro avg       0.68      0.64      0.64      
         601\nweighted avg       0.69      0.69      0.67      
         601\n'
         
         
         69% Accuracy obtained  overall.
'''


###------------------------ Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(aff, test_size = 0.3) # 30% test data


# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('naffairs ~ kids + vryunhap + unhap + avgmarr + hapavg' +
                          ' + vryhap + antirel  +notrel + slghtrel +  smerel + vryrel'+
                          '+ yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5 + yrsmarr6',data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = model.predict(test_data)    #predict(X_test)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(181)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data["test_pred"], test_data['naffairs'])
confusion_matrix

'''
naffairs    0   1
test_pred        
0.0        96  16
1.0        38  31
'''

accuracy_test = (31 + 96)/(181) 
accuracy_test   # 0.7016574585635359 

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["naffairs"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["naffairs"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test
#0.7176881549698316

# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(420)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['naffairs'])
confusion_matrx

'''
naffairs      0   1
train_pred         
0.0         227  40
1.0          90  63
'''


accuracy_train = (63 + 227)/(420)
print(accuracy_train)
# 0.6904761904761905
                   