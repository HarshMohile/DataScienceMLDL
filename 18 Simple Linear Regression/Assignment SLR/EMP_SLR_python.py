# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import matplotlib.pyplot as plt
import seaborn as sns

emp = pd.read_csv("D:\\360Assignments\\Submission\\18 Simple Linear Regression\\Assignment SLR\\emp_data.csv")

## NA or null
emp.head()
emp.describe()

## isnull() comes from pd and  np,
emp.isnull().values.any()
emp.isnull().sum()

np.any(np.isnan(emp))
np.all(np.isfinite(emp))

#np.any(np.isnull(emp)) doesnt exist


## Data vis
emp.columns
#Index(['empary_hike', 'Churn_out_rate'], dtype='object')


plt.scatter(x='Salary_hike', y= 'Churn_out_rate', data=emp, cmap='viridis')
plt.xlabel("Salary_hike")
plt.ylabel("Churn_out_rate")
# More the Churn out rate the less the empary_hike is noticed. Giving a downward curve.

plt.boxplot(emp['Salary_hike'])
plt.boxplot(emp['Churn_out_rate'])

sns.jointplot(x='Salary_hike', y= 'Churn_out_rate', data=emp, cmap='viridis')

sns.heatmap(emp.corr(), annot=True)
# 0.91 for both hike <--> churn to each other

# Import library
import statsmodels.formula.api as smf


################################# Simple Linear Regression   'AT ~ Waist'
## results = sm.OLS(y, X).fit()
model = smf.ols('Salary_hike ~ Churn_out_rate', data = emp).fit()
model.summary()



pred1 = model.predict(pd.DataFrame(emp['Churn_out_rate']))  # X_test feeding yurExp to get predict empary

# Regression Line
plt.scatter(x=emp['Churn_out_rate'], y=emp['Salary_hike']) # scatter points 
plt.plot(emp['Churn_out_rate'], pred1, "r")      # regression line .best fit line( actual Yrs EXp , Predicted emp from pred1)
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = emp['Salary_hike'] - pred1   # Residual individually (y-ypred)
res_sqr1 = res1 * res1       # SSE        (y-ypred)^2
mse1 = np.mean(res_sqr1)              #mean of sse
rmse1 = np.sqrt(mse1)          #root mean sse
rmse1
 ### ---> 35.89263510276639 as RMSE
 
 
  ################################ Model building on Transformed Data
# Log Transformation
# x = log(YrsEXp); y = Sal


#emp['Churn_out_rate']
#emp['Salary_hike']

plt.scatter(x = np.log(emp['Churn_out_rate']), y = emp['Salary_hike'], color = 'brown')

np.corrcoef(np.log(emp['Churn_out_rate']),emp['Salary_hike']) #correlation
'''
array([[ 1.        , -0.93463607],
       [-0.93463607,  1.        ]])   Since its is going downward it in -ve
'''

model2 = smf.ols('Salary_hike ~ np.log(Churn_out_rate)', data = emp).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(emp['Churn_out_rate'])) 
# testing the model by test data i.e chrnrate   to predict Salary_hike

# Regression Line
plt.scatter(np.log(emp['Churn_out_rate']), emp['Salary_hike'])
plt.plot(np.log(emp['Churn_out_rate']), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = emp['Salary_hike'] - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2
###  31.06952064024449 --> as RMSE



#### Exponential transformation
# x = waist; y = log(at)

#sal['Churn_out_rate']
#sal['Salary_hike']

plt.scatter(x = emp['Churn_out_rate'], y = np.log(emp['Salary_hike']), color = 'orange')
np.corrcoef(emp['Churn_out_rate'], np.log(emp['Salary_hike'])) #correlation

model3 = smf.ols('np.log(Salary_hike) ~ Churn_out_rate', data = emp).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(emp['Churn_out_rate']))
pred3_at = np.exp(pred3)   # because its Exponential Transformation
pred3_at

# Regression Line
plt.scatter(emp.Churn_out_rate, np.log(emp.Salary_hike),cmap="viridis")
plt.plot(emp.Churn_out_rate, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 =emp.Salary_hike - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#34.268549806744616 --> as RMSE when using Polynomial transformation


###################
# The best model

# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3 ])}
table_rmse = pd.DataFrame(data)
table_rmse

'''
       MODEL       RMSE
0        SLR  35.892635
1  Log model  31.069521
2  Exp model  34.268550

Log tranformation performed better at giving less RMSE for this dataset
'''

from sklearn.model_selection import train_test_split

train, test = train_test_split(emp, test_size = 0.2)


# Performing tran test split and model building using Log transformtion
finalmodel = smf.ols('Salary_hike ~ np.log(Churn_out_rate)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))   
pred_test_Sal = np.exp(test_pred)
pred_test_Sal



# Regression Line
plt.scatter(np.log(emp['Churn_out_rate']), emp['Salary_hike'])
plt.plot(test, pred_test_Sal, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

 


