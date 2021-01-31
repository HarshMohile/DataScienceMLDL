# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import matplotlib.pyplot as plt
import seaborn as sns


cal = pd.read_csv("D:\\360Assignments\\Submission\\18 Simple Linear Regression\\Assignment SLR\\calories_consumed.csv")

cal.describe()

# Check null or nan values
cal.isnull().values.any()
cal.isnull().sum().sum()

np.any(np.isnan(cal))
np.all(np.isfinite(cal))

cal.dropna()
# No NA or Null values present

## Data vis
cal.columns


plt.scatter(x='Weight gained (grams)', y= 'Calories Consumed', data=cal, cmap='viridis')
# it forms a linearcurve . increase in cal increases weigth.

plt.boxplot(cal['Weight gained (grams)'])
plt.boxplot(cal['Calories Consumed'])

sns.jointplot(x='Weight gained (grams)', y= 'Calories Consumed', data=cal, cmap='viridis')


# correlation
np.corrcoef(cal['Weight gained (grams)'], cal['Calories Consumed']) 
'''
array([[1.        , 0.94699101],
       [0.94699101, 1.        ]])
'''

# Import library
import statsmodels.formula.api as smf



cal.rename(columns={'Weight gained (grams)':'Weigth'},inplace=True)
cal.rename(columns={'Calories Consumed': 'Calor_consumed'},inplace=True)

################################# Simple Linear Regression   'AT ~ Waist'
## results = sm.OLS(y, X).fit()
model = smf.ols('Weigth ~ Calor_consumed', data = cal).fit()
model.summary()

'''
[2] The condition number is large, 8.28e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
'''

pred1 = model.predict(pd.DataFrame(cal['Calor_consumed']))  # X_test

# Regression Line
plt.scatter(cal.Calor_consumed, cal.Weigth) # scatter points 
plt.plot(cal.Calor_consumed, pred1, "r")             # regression line .best fit line( actual , Predicted)
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = cal.Weigth - pred1   # Residual individually (y-ypred)
res_sqr1 = res1 * res1       # SSE        (y-ypred)^2
mse1 = np.mean(res_sqr1)              #mean of sse
rmse1 = np.sqrt(mse1)          #root mean sse
rmse1
 ### ---> 103.30250194726935 aas RMSE


################################ Model building on Transformed Data
# Log Transformation
# x = log(Calories); y = weigth


plt.scatter(x = np.log(cal['Calor_consumed']), y = cal['Weigth'], color = 'brown')

np.corrcoef(np.log(cal.Calor_consumed), cal.Weigth) #correlation
'''
array([[1.        , 0.89872528],
       [0.89872528, 1.        ]])
'''

model2 = smf.ols('Weigth ~ np.log(Calor_consumed)', data = cal).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(cal['Calor_consumed']))

# Regression Line
plt.scatter(np.log(cal.Calor_consumed), cal.Weigth)
plt.plot(np.log(cal.Calor_consumed), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = cal.Weigth - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2
### 141.0053816942511


################### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = cal['Calor_consumed'], y = np.log(cal['Weigth']), color = 'orange')

np.corrcoef(cal.Calor_consumed, np.log(cal.Weigth)) #correlation
'''
array([[1.        , 0.93680369],
       [0.93680369, 1.        ]])
'''

model3 = smf.ols('np.log(Weigth) ~ Calor_consumed', data = cal).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(cal['Calor_consumed']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(cal.Calor_consumed, np.log(cal.Weigth))
plt.plot(cal.Calor_consumed, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = cal.Weigth - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3
##### 118.04515720118044

################### Polynomial transformation
# x = Calor_consumed; x^2 = Calor_consumed*Calor_consumed; y = log(Weigth)

model4 = smf.ols('np.log(Weigth) ~ Calor_consumed + I(Calor_consumed*Calor_consumed)', data = cal).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(cal))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = cal.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = cal.iloc[:, 1].values


plt.scatter(cal.Calor_consumed, np.log(cal.Weigth))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = cal.Weigth - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

#### 117.4145001310951

# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse



'''
        MODEL        RMSE
0         SLR  103.302502
1   Log model  141.005382
2   Exp model  118.045157
3  Poly model  117.414500

'''

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(cal, test_size = 0.2)

##finalmodel = smf.ols('np.log(Weigth) ~ Calor_consumed + I(Calor_consumed*Calor_consumed)', data = train).fit()
finalmodel = smf.ols('Weigth ~ Calor_consumed',data=train ).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))   #Predict on test data predict(whole_test)
pred_test_Weigth = np.exp(test_pred)
pred_test_Weigth

'''
10    1.933098e+16
6     1.030439e-35
2              inf
dtype: float64
'''

# Model Evaluation on Test data
test_res = test.Weigth - pred_test_Weigth
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Weigth = np.exp(train_pred)
pred_train_Weigth

# Model Evaluation on train data
train_res = train['Weigth'] - pred_train_Weigth
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

