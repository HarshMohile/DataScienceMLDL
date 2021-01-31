import pandas as pd
import numpy as np
import scipy
from scipy import stats 
import statsmodels.api as sm


fant= pd.read_csv("D:\\360Assignments\\Submission\\17 Hypothesis Testing\\Fantaloons.csv")

'''
lab_tat_updated
CustomerOrderform
Fantaloons
BuyerRatio
Cutlets
'''

# Check whether the 
# Ho --> No change in stores in which males go into  (by day)
# H1 --> There is significant  change in % for males v females walking in different stores

fant.columns

fant.describe()
'''
       Weekdays Weekend
count       400     400
unique        2       2
top      Female  Female
freq        287     233
'''

fant['Weekdays'].value_counts()
#Female    287
#Male      113

fant['Weekend'].value_counts()
#Female    233
#Male      167
# variance and normality doesnt support Str values

from statsmodels.stats.proportion import proportions_ztest

tab1 = fant['Weekdays'].value_counts()
tab1
tab2 = fant['Weekend'].value_counts()
tab2

# crosstable table
pd.crosstab(fant['Weekdays'], fant['Weekend'])

count = np.array([120, 58])
nobs = np.array([113, 287])

stats, pval = proportions_ztest(count, nobs, alternative = 'two-sided') 
print(pval) # Pvalue 1.0055944209330287e-54

stats, pval = proportions_ztest(count, nobs, alternative = 'larger')
print(pval)  # 5.027972104665143e-55

# We reject Null hypothesis so there is definetly change in sales happening from centersboth Male and Female on Weekdays and WeekEnds 





