
# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import matplotlib.pyplot as plt
import seaborn as sns

sal = pd.read_csv("D:\\360Assignments\\Submission\\18 Simple Linear Regression\\Assignment SLR\\Salary_Data.csv")


sal.head()
sal.describe()

## isnull() comes from pd and  np,
sal.isnull().values.any()
sal.isnull().sum()

np.any(np.isnan(sal))
np.all(np.isfinite(sal))

## Data vis
sal.columns
#Index(['YearsExperience', 'Salary'], dtype='object')


plt.scatter(x='YearsExperience', y= 'Salary', data=sal, cmap='viridis')
# there is a linear curve but not  like a straight line but noticable

plt.boxplot(sal['YearsExperience'])
plt.boxplot(sal['Salary'])

sns.jointplot(x='YearsExperience', y= 'Salary', data=sal, cmap='viridis')

sns.heatmap(sal.corr(), annot=True)

# Import library
import statsmodels.formula.api as smf


################################# Simple Linear Regression   'AT ~ Waist'
## results = sm.OLS(y, X).fit()
model = smf.ols('Salary ~ YearsExperience', data = sal).fit()
model.summary()



pred1 = model.predict(pd.DataFrame(sal['YearsExperience']))  # X_test feeding yurExp to get predict Salary

# Regression Line
plt.scatter(x=sal['YearsExperience'], y=sal['Salary']) # scatter points 
plt.plot(sal['YearsExperience'], pred1, "r")      # regression line .best fit line( actual Yrs EXp , Predicted Sal from pred1)
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = sal['Salary'] - pred1   # Residual individually (y-ypred)
res_sqr1 = res1 * res1       # SSE        (y-ypred)^2
mse1 = np.mean(res_sqr1)              #mean of sse
rmse1 = np.sqrt(mse1)          #root mean sse
rmse1
 ### ---> 5592.043608760661 aas RMSE
 
 
 ################################ Model building on Transformed Data
# Log Transformation
# x = log(YrsEXp); y = Sal


#sal['YearsExperience']
#sal['Salary']

plt.scatter(x = np.log(sal['YearsExperience']), y = sal['Salary'], color = 'brown')

np.corrcoef(np.log(sal['YearsExperience']),sal['Salary']) #correlation
'''
array([[1.        , 0.92406108],
       [0.92406108, 1.        ]])
'''

model2 = smf.ols('Salary ~ np.log(YearsExperience)', data = sal).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(sal['YearsExperience'])) 
# testing the model by test data i.e Srtime to predict Deliverytime

# Regression Line
plt.scatter(np.log(sal['YearsExperience']), sal['Salary'])
plt.plot(np.log(sal['YearsExperience']), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = sal['Salary'] - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2
###  10302.893706228302






#### Exponential transformation
# x = waist; y = log(at)

#sal['YearsExperience']
#sal['Salary']

plt.scatter(x = sal['YearsExperience'], y = np.log(sal['Salary']), color = 'orange')
np.corrcoef(sal['YearsExperience'], np.log(sal['Salary'])) #correlation

model3 = smf.ols('np.log(Salary) ~ YearsExperience', data = sal).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(sal['YearsExperience']))
pred3_at = np.exp(pred3)   # because its Exponential Transformation
pred3_at

# Regression Line
plt.scatter(sal.YearsExperience, np.log(sal.Salary),cmap="viridis")
plt.plot(sal.YearsExperience, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 =sal.Salary - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#7213.23507662012 --> as RMSE


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience)', data = sal).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(sal))  ## Whole of dataset as Test data for predictions
pred4_at = np.exp(pred4)                    ### Then take Expo of it 
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2)

''' #### X and y
X = sal.iloc[:, 0:1].values ## 0:n-1 we only take YearsExperince in X
y = sal.iloc[:, 1].values
X
y
X_poly = poly_reg.fit_transform(X)

'''

plt.scatter(sal.YearsExperience, np.log(sal.Salary))
plt.plot(sal.YearsExperience, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = sal.Salary - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


## 5391.08158269357 --> as RMSE  for polynomial

###################
# The best model

# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

'''
        MODEL          RMSE
0         SLR   5592.043609
1   Log model  10302.893706
2   Exp model   7213.235077
3  Poly model   5391.081583


 Polynomial performed better with less RMSE
'''

from sklearn.model_selection import train_test_split

train, test = train_test_split(sal, test_size = 0.2)

finalmodel = smf.ols('np.log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))   
pred_test_Sal = np.exp(test_pred)
pred_test_Sal

# Model Evaluation on Test data
test_res = test.Salary  - pred_test_Sal   ## Actual - ypredTest
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse
## 5849.0417615844335 --> RMSE fro Test Data

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Sal = np.exp(train_pred)
pred_train_Sal

train_res = train.Salary - pred_train_Sal
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

### 5235.602483490605 --> less training error as low RMSE in Train data . A case of slight Overfitting
