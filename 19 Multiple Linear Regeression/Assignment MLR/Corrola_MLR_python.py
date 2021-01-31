# Multilinear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# loading the data
toy = pd.read_csv("D:\\360Assignments\\Submission\\"
                +"19 Multiple Linear Regeression\\Assignment MLR\\ToyotaCorolla.csv",encoding='latin1')

# predict the Price  variable by these other independent variables 

toy.head()
toy.describe()


toy.columns
'''
Index(['Id', 'Model', 'Price', 'Age_08_04', 'Mfg_Month', 'Mfg_Year', 'KM',
       'Fuel_Type', 'HP', 'Met_Color', 'Color', 'Automatic', 'cc', 'Doors',
       'Cylinders', 'Gears', 'Quarterly_Tax', 'Weight', 'Mfr_Guarantee',
       'BOVAG_Guarantee', 'Guarantee_Period', 'ABS', 'Airbag_1', 'Airbag_2',
       'Airco', 'Automatic_airco', 'Boardcomputer', 'CD_Player',
       'Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio',
       'Mistlamps', 'Sport_Model', 'Backseat_Divider', 'Metallic_Rim',
       'Radio_cassette', 'Tow_Bar'],
      dtype='object')
'''
# Data cleaning
toy1= toy[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]

toy1.isnull().values.any()

plt.bar(height = toy1['Price'],x=toy1['Quarterly_Tax'])
plt.hist(toy1['Age_08_04']) #histogram
plt.boxplot(toy1['Price']) #boxplot
plt.scatter(x=toy1['Price'], y=toy1['Quarterly_Tax'])




# heatmap for corr
sns.heatmap(toy1.corr(),annot=True)

# PAirplot to see the relation y and x
sns.pairplot(toy1)

# Jointplot
import seaborn as sns
sns.jointplot(x=toy1['HP'], y=toy1['cc'])
# No correlation between Rnd Spend  for each state

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(toy1['Price'], dist = "norm", plot = pylab) # outliers are found at the top of the probplot
plt.show()

toy1.corr()


# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm


         
ml1 = smf.ols('Price ~ Age_08_04+ KM+ HP + cc + Doors + Gears +  Quarterly_Tax + Weight', data = toy1).fit() # regression model

# Summary
ml1.summary()
# p-values for Doors 0.968 and cc 0.179 are more than 0.05


sm.graphics.influence_plot(ml1)  # Distance Calc using Cooks distance------------------------ first time plotting
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row

toy_new = toy1.drop(toy1.index[[80]])


# Now again check the model after removing influencing rows 
# Preparing model                  
ml_new = smf.ols('Price ~ Age_08_04+ KM+ HP + cc + Doors + Gears +  Quarterly_Tax + Weight', data = toy_new).fit()    

# Summary
ml_new.summary()
## P  values for Administration and MarketingSpend are now less than before 
# Earlier  it was 60% and 20% for both variables

ml_new_rsq = smf.ols('Price ~ Age_08_04+ KM+ HP + cc + Doors + Gears +  Quarterly_Tax + Weight', data = toy_new).fit().rsquared  

vif_model = 1/1-ml_new_rsq # 0.13062095113441818 for the entire model

#--------------------------------------VIF-----------------------------------------
# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables

from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


y,X = dmatrices('Price ~ Age_08_04+ KM+ HP + cc + Doors + Gears +  Quarterly_Tax + Weight', data = toy_new,return_type='dataframe')  

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()  # empty dataframe

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(9)]
vif["features"] = X.columns


# X.shape[1] return  9 : number of columns
vif.head()

#-------------------------------------------------------------------------------------------------

#///////////////////////////////////////  QQ plot using Seaborn and stats.probplot //////////////
# Q-Q plot of Residuals  using SEABORN
res = ml_new.resid
sm.qqplot(res)

plt.show()

# Q-Q plot of Resiuals using STATS.PROBPLOT
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()


# Prediction
pred = ml_new.predict(toy_new)

# Residuals vs Fitted plot
sns.residplot(x = pred, y = toy_new['Price'], lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(ml_new) # ------------------------------ second time plotting


#//////////////////////////////////////////////DATA PARTITION SPLIT ////////////////////////////////////////////

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
train, test = train_test_split(toy_new, test_size = 0.2) # 20% test data


toy_new.isnull().values.any()
toy_new.isnull().sum()


# preparing the model on train data 
model_train = smf.ols("Price ~ Age_08_04+ KM+ HP + cc + Doors + Gears +  Quarterly_Tax + Weight", data = train).fit()

# prediction on test data set 
test_pred = model_train.predict(test)   # pred =predict(X_test)



# test residual values Predicted - Actual .PREDICTION ON TEST DATA 
test_resid = test_pred - test['Price']


# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse
# 1261.3610332820308 as RMSE for test data

# train_data prediction
train_pred = model_train.predict(train)

# train residual values   Predicted - Actual .PREDICTION ON TRAIN DATA 
train_resid  = train_pred - train['Price']
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
# 1321.544320570788 Training Error is  more than test data error




