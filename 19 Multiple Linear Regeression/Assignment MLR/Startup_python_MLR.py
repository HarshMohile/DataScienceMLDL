# Multilinear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# loading the data
strt = pd.read_csv("D:\\360Assignments\\Submission\\19 Multiple Linear Regeression\\Assignment MLR\\50_Startups.csv")

strt.head()

strt.head()
strt.describe()

## isnull() comes from pd and  np,
strt.isnull().values.any()
strt.isnull().sum()

strt.columns
#Index(['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit'], dtype='object')

## New  york has the highest plot of all countries in terms of Marketing and RnD
# R&D Spend
plt.bar(height = strt['R&D Spend'],x=strt['State'])
plt.hist(strt['R&D Spend']) #histogram
plt.boxplot(strt['R&D Spend']) #boxplot

# marketing Spend
plt.bar(height = strt['Marketing Spend'], x=strt['State'])
plt.hist(strt['Marketing Spend']) #histogram
plt.boxplot(strt['Marketing Spend']) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=strt['R&D Spend'], y=strt['State'])
# No correlation between Rnd Spend  for each state

# Scatter
plt.scatter(x='State', y= 'R&D Spend', data=strt, cmap='viridis')

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(strt['Marketing Spend'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(strt['Marketing Spend'], dist = "norm", plot = pylab)
plt.show()


# Scatter plot between the variables along with histograms

sns.pairplot(strt.iloc[:, :])

#  Good Correlation  between  RnD Spent and  Profit earned                         
# Correlation matrix 
strt.corr()

# we see there exists High collinearity between input variables especially between
# [RnD Spend & Marketing ] so there exists collinearity problem

# X= Rnd ,Merketing , Admin 
# y= Profit
strt.columns
#Index(['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit'], dtype='object')

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

#Rename a col with whitespace
strt.rename(columns={"Marketing Spend":"MarketingSpend"},inplace=True)
strt.rename(columns={"R&D Spend":"RnDSpend"},inplace=True)
         
ml1 = smf.ols('Profit ~ RnDSpend + Administration + MarketingSpend', data = strt).fit() # regression model

# Summary
ml1.summary()
# p-values for Administration, MarketingSpend are more than 0.05


sm.graphics.influence_plot(ml1)  # Distance Calc using Cooks distance
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row

strt_new = strt.drop(strt.index[[49]])
strt_new = strt.drop(strt.index[[48]])

# Now again check the model after removing influencing rows 
# Preparing model                  
ml_new = smf.ols('Profit ~ RnDSpend + Administration + MarketingSpend', data = strt_new).fit()    

# Summary
ml_new.summary()
## P  values for Administration and MarketingSpend are now less than before 
# Earlier  it was 60% and 20% for both variables



#--------------------------------------VIF-----------------------------------------
# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_AdminS = smf.ols('Administration ~ RnDSpend + MarketingSpend', data = strt).fit().rsquared  
vif_ad = 1/(1 - rsq_AdminS) 

rsq_rnd = smf.ols('RnDSpend ~ Administration + MarketingSpend', data = strt).fit().rsquared  
vif_rnd = 1/(1 - rsq_rnd)

rsq_mrt = smf.ols('MarketingSpend ~ Administration + RnDSpend ', data = strt).fit().rsquared  
vif_mrt = 1/(1 - rsq_mrt) 



# Rsquared for Administration 0.14900208239517543 and VIF 1.1750910070550455
#Rsquare for RndSpend  0.5949618224573936 and VIF 2.4689030699947017
# Rsquare for Marketing 0.5702202685282503 and VIF  2.3267732905308773
# none of them have collinearity now after removing those 48 and 46 indexes from dataset


# Storing vif values in a data frame
dict_1 = {'Variables':['MarketingSpend', 'Administration', 'RnDSpend'], 'VIF':[vif_mrt, vif_ad, vif_rnd]}

type(dict_1)
Vif_df = pd.DataFrame(dict_1)  
Vif_df

#-------------------------------------------------------------------------------------------------
# Final model (No need of removing  any column)------------------------------------------
final_ml = smf.ols('Profit ~ RnDSpend + Administration + MarketingSpend', data = strt_new).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(strt_new)


#///////////////////////////////////////  QQ plot using Seaborn and stats.probplot //////////////
# Q-Q plot of Residuals  using SEABORN
res = final_ml.resid
sm.qqplot(res)

plt.show()

# Q-Q plot of Resiuals using STATS.PROBPLOT
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = strt_new['Profit'], lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)
#//////////////////////////////////////////////DATA PARTITION SPLIT ////////////////////////////////////////////

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
train, test = train_test_split(strt_new, test_size = 0.2) # 20% test data


strt_new.isnull().values.any()
strt_new.isnull().sum()


# preparing the model on train data 
model_train = smf.ols("Profit ~ RnDSpend + Administration + MarketingSpend", data = train).fit()

# prediction on test data set 
test_pred = model_train.predict(test)   # pred =predict(X_test)



# test residual values Predicted - Actual  
test_resid = test_pred - test['Profit']
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse
# 8444.532150505838 as RMSE for test data

# train_data prediction
train_pred = model_train.predict(train)

# train residual values 
train_resid  = train_pred - train['Profit']
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
# 8810.210145423645 Training Error





