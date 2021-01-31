# LR Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

comp = pd.read_csv("D:\\360Assignments\\Submission\\21 Lasso Ridge Regression\\Assignment LRR\\Computer_Data.csv")

comp.head()

comp.describe()

## isnull() comes from pd and  np,
comp.isnull().values.any()
comp.isnull().sum()

comp.columns

#data preprocessing  & cleaning
comp= comp.drop('Unnamed: 0',axis=1)
# Label encoding by categorical data
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
comp["cd"] = lb.fit_transform(comp["cd"])
comp["multi"] = lb.fit_transform(comp["multi"])
comp["premium"] = lb.fit_transform(comp["premium"])



# EDA 

plt.bar(height = comp['speed'], x= comp['trend'])
plt.hist(comp['screen']) #histogram  # majority of users prefer 14 inch monitor 

# iterate over all cols aas boxplot
for i in comp.columns:
    plt.figure()
    comp.boxplot([i]) #boxplot
    
# Correlation
sns.heatmap(comp.corr(), annot=True)

# Implementing the model using Linear Regression to know the reason of choosing LR Regerssion

X = comp.drop('price',axis=1)
y=comp['price']

import statsmodels.formula.api as smf

model_lm = smf.ols("price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend", data = comp).fit()
model_lm.summary()
# all are getting 0 as p values so that means we reject null hypthesis and they contribute to my predictor and response


pred = model_lm.predict(comp)
# Error
resid  = pred - comp['price']
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse
# 275.12981886387195

#---------------------------------------------------LR REGRESSION---------------------------------------------


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
lasso.score(X,y)     #0.7715882298605266

# RMSE
np.sqrt(np.mean((pred_lasso - y)**2))   #277.55822974069537


# we need to find the optimal value of alpha (hypertuning parameter ) for our dataset
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
print('MAE: %.3f' % results.best_score_)   #MAE: -203.488
print('Config: %s' % results.best_params_)  #Config: {'alpha': 0.11}

# choosing apha as 0.11 and model it again
lasso_Adj = Lasso(alpha = 0.11, normalize = True)

lasso_Adj.fit(X, y)

# Coefficient values for all independent variables#
lasso_Adj.coef_
lasso_Adj.intercept_
lasso_Adj.alpha

#Prdeictions
pred_lasso = lasso.predict(X)

# LASSO  ADJUSTED  R SCORE
lasso.score(X,y)     #0.7715882298605266  -->0.7715882298605266

# RMSE
np.sqrt(np.mean((pred_lasso - y)**2))   #277.55822974069537

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

ridge_reg.best_params_   # {'alpha': 1}
ridge_reg.best_score_    # 0.7743192743532367

ridge_pred = ridge_reg.predict(X)

# Adjusted r-square#
ridge_reg.score(X, y)     #0.7755673168845666

# RMSE
np.sqrt(np.mean((ridge_pred - y)**2))   # 17369.486541504262

 # Applying alpha value as 1
# choosing apha as 0.11 and model it again
ridge_Adj = Ridge(alpha = 1, normalize = True)

ridge_Adj.fit(X, y)

# Coefficient values for all independent variables#
ridge_Adj.coef_
ridge_Adj.intercept_
ridge_Adj.alpha

#Prdeictions
pred_ridge = ridge_Adj.predict(X)

# Ridge  ADJUSTED  R SCORE
ridge_Adj.score(X,y)     #0.5350889766460389

# RMSE
np.sqrt(np.mean((pred_ridge - y)**2))   #395.9859628078518
# Ridge model performed worse than Lasso did even with GridSearch where we evaluated  the perfect value for alpha
 
