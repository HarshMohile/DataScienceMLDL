
# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import matplotlib.pyplot as plt
import seaborn as sns

deltime = pd.read_csv("D:\\360Assignments\\Submission\\18 Simple Linear Regression\\Assignment SLR\\delivery_time.csv")

deltime.describe()

deltime.describe()

# Check null or nan values
deltime.isnull().values.any()
deltime.isnull().sum()

np.any(np.isnan(deltime))
np.all(np.isfinite(deltime))

deltime.dropna()
# No NA or Null values present

## Data vis
deltime.columns
#Index(['Delivery Time', 'Sorting Time'], dtype='object')


plt.scatter(x='Sorting Time', y= 'Delivery Time', data=deltime, cmap='viridis')
# there is a linear curve but not  like a straight line but noticable

plt.boxplot(deltime['Sorting Time'])
plt.boxplot(deltime['Delivery Time'])

sns.jointplot(x='Sorting Time', y= 'Delivery Time', data=deltime, cmap='viridis')


# correlation
np.corrcoef(deltime['Sorting Time'], deltime['Delivery Time']) 
'''
array([[1.        , 0.82599726],
       [0.82599726, 1.        ]])
'''

# Import library
import statsmodels.formula.api as smf



deltime.rename(columns={'Sorting Time':'Srtime'},inplace=True)
deltime.rename(columns={'Delivery Time': 'Dltime'},inplace=True)

################################# Simple Linear Regression   'AT ~ Waist'
## results = sm.OLS(y, X).fit()
model = smf.ols('Dltime ~ Srtime', data = deltime).fit()
model.summary()



pred1 = model.predict(pd.DataFrame(deltime['Srtime']))  # X_test

# Regression Line
plt.scatter(deltime.Srtime, deltime.Dltime) # scatter points 
plt.plot(deltime.Srtime, pred1, "r")             # regression line .best fit line( actual , Predicted)
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = deltime.Dltime - pred1   # Residual individually (y-ypred)
res_sqr1 = res1 * res1       # SSE        (y-ypred)^2
mse1 = np.mean(res_sqr1)              #mean of sse
rmse1 = np.sqrt(mse1)          #root mean sse
rmse1
 ### ---> 2.7916503270617654 aas RMSE


################################ Model building on Transformed Data
# Log Transformation
# x = log(Calories); y = weigth


plt.scatter(x = np.log(deltime['Srtime']), y = deltime['Dltime'], color = 'brown')

np.corrcoef(np.log(deltime['Srtime']), deltime['Dltime']) #correlation
'''
array([[1.        , 0.83393253],
       [0.83393253, 1.        ]])
'''

model2 = smf.ols('Dltime ~ Srtime', data = deltime).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(deltime['Srtime'])) 
# testing the model by test data i.e Srtime to predict Deliverytime

# Regression Line
plt.scatter(np.log(deltime.Srtime), deltime.Dltime)
plt.plot(np.log(deltime.Srtime), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = deltime.Dltime - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2
###  2.7916503270617654



# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model",]), "RMSE":pd.Series([rmse1, rmse2])}
table_rmse = pd.DataFrame(data)
table_rmse



'''
       MODEL     RMSE
0        SLR  2.79165
1  Log model  2.79165

'''
# both Log and SLR have the same RMSE so we choose the default one 
###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(deltime, test_size = 0.2)

##finalmodel = smf.ols('np.log(Weigth) ~ Calor_consumed + I(Calor_consumed*Calor_consumed)', data = train).fit()
finalmodel = smf.ols('Dltime ~ Srtime',data=train ).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))   #Predict on test data predict(whole_test)
pred_test_Dltime = np.exp(test_pred)
pred_test_Dltime



# Model Evaluation on Test data
test_res = test.Dltime - pred_test_Dltime
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse
#### --> 12002150006.1002 for Test Data 

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Dltime = np.exp(train_pred)
pred_train_Dltime

# Model Evaluation on train data
train_res = train['Dltime'] - pred_train_Dltime
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
#### 9603211653.488464 --> A lot of RMSE in Train data but better at Test data

