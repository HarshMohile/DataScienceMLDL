# LR Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

strt = pd.read_csv("D:\\360Assignments\\Submission\\21 Lasso Ridge Regression\\Assignment LRR\\50_Startups.csv")

strt.head()

strt.head()
strt.describe()

## isnull() comes from pd and  np,
strt.isnull().values.any()
strt.isnull().sum()

strt.columns

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


# Correlation matrix 
a = strt.corr()
a

#Rename a col with whitespace
strt.rename(columns={"Marketing Spend":"MarketingSpend"},inplace=True)
strt.rename(columns={"R&D Spend":"RnDSpend"},inplace=True)


# Replace 0 with mean 
strt = strt.mask(strt==0).fillna(strt.mean())

## State for State Column

strt1 =pd.get_dummies(strt)

strt1.columns
'''
Index(['RnDSpend', 'Administration', 'MarketingSpend', 'Profit',
       'State_California', 'State_Florida', 'State_New York'],
      dtype='object')
'''
#renaming New york col
strt1.rename(columns={"State_New York":"State_NewYork"},inplace=True)

# Implementing the model using Linear Regression to know the reason of choosing LR Regerssion

import statsmodels.formula.api as smf
model_lm = smf.ols("Profit ~ RnDSpend + Administration + MarketingSpend + State_California +State_Florida + State_NewYork", data = strt1).fit()
model_lm.summary()

# Prediction using whole dataset (No X_test data .direct whole dataset)
pred = model_lm.predict(strt1)
# Error
resid  = pred - strt['Profit']
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

##17302.475190519068
# reason behind using LRR is to desentized our data .Right there is collinearity and 
#causiing our X features difference actually make our prediction sensitive

#---------------------------------------------------LR REGRESSION---------------------------------------------

#X= strt1.iloc[:,0:4]

X=strt1.drop('Profit',axis=1)
y=strt1['Profit']


###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(X, y)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_
lasso.alpha

#Prdeictions
pred_lasso = lasso.predict(X)

# LASSO  ADJUSTED  R SCORE
lasso.score(X,y)     #0.8119613558983276

# RMSE
np.sqrt(np.mean((pred_lasso - strt1['Profit'])**2))   #17302.475262366916


# we need to find the optimal value of alpha (hypertuning prarameter ) for our dataset
#-----------------------------------------------------------------------------------------------------
#------------------------------ USING GRIDSEARCHCV TO IMPROVE LASSO  PREVIOUS RESULTS-----------
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold


# define model evaluation method

lasso # our model name
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['alpha'] = np.arange(0, 20, 0.01)

search = GridSearchCV(lasso, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('MAE: %.3f' % results.best_score_)   #MAE: -12379.827
print('Config: %s' % results.best_params_)  #Config: {'alpha': 19.990000000000002}

# choosing apha as 20 and model it again
lasso_Adj = Lasso(alpha = 20, normalize = True)

lasso_Adj.fit(X, y)

# Coefficient values for all independent variables#
lasso_Adj.coef_
lasso_Adj.intercept_
lasso_Adj.alpha

#Prdeictions
pred_lasso = lasso.predict(X)

# LASSO  ADJUSTED  R SCORE
lasso.score(X,y)     #0.8119613558983276  -->0.8119613558983276

# RMSE
np.sqrt(np.mean((pred_lasso - strt1['Profit'])**2))   #17302.475262366916

 # not much difference after changing alpha .So in future we will keep alpha as 0.13
 
 #--------------------------------------------------------------------------------------------------
 #----------------------------------------- RIDGE REGRESSION ---------------------------------------
 # Ridge Regression USING  GRIDSEARCH CV 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# OUR MODEL RIDGE
ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv =cv)

ridge_reg.fit(X, y)

ridge_reg.best_params_   # {'alpha': 20}
ridge_reg.best_score_    # 0.7124796021479034

ridge_pred = ridge_reg.predict(X)

# Adjusted r-square#
ridge_reg.score(X, y)     #0.8105020143055895

# RMSE
np.sqrt(np.mean((ridge_pred - y)**2))   # 17369.486541504262


 
 
