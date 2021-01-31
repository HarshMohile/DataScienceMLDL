import pandas as pd
import numpy as np
import scipy
from scipy import stats 
import statsmodels.api as sm

fba= pd.read_csv("D:\\360Assignments\\Submission\\17 Hypothesis Testing\\Cutlets.csv")

'''
lab_tat_updated
CustomerOrderform
Fantaloons
BuyerRatio
Cutlets
'''
## Dropping Nan /null values
np.any(np.isnan(fba))  

fba=fba.dropna()

# Check whether the diameter of both the donuts are same or not
# Ho --> No difference in diameter
# H1 --> There is significant difference in diameter of 2 donuts

# Usng Z test as number of samples are more than 30 

fba.columns
scipy.stats.zscore(fba['Unit A'], axis=0)

# Normality Test
stats.shapiro(fba['Unit A']) # Shapiro Test
#ShapiroResult(statistic=0.9649458527565002, pvalue=0.3199819028377533) >0.05 Accept Null hypothesis

# Variance test
scipy.stats.levene(fba['Unit A'], fba['Unit B'])
#LeveneResult(statistic=0.665089763863238, pvalue=0.4176162212502553) >0.05 Accept Null hypothesis
help(scipy.stats.levene)

# Chi Square Test
count = pd.crosstab(fba['Unit A'], fba['Unit B'])
count
Chisquares_results = scipy.stats.chi2_contingency(count)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square

#[['Test Statistic', 'p-value'], [1190.0000000000005, 0.23757566509339403]] .0.2375 > 0.05 .Accept Null hypothesis

#  Conclusion::No difference in diameter of donuts




