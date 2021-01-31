import pandas as pd
import numpy as np
import scipy
from scipy import stats 
import statsmodels.api as sm

sales= pd.read_csv("D:\\360Assignments\\Submission\\17 Hypothesis Testing\\BuyerRatio.csv")

'''
lab_tat_updated
CustomerOrderform
Fantaloons
BuyerRatio
Cutlets
'''

sales.columns

# Check whether the diameter of both the donuts are same or not
# Ho --> No difference in Proportion
# H1 --> There is significant difference Proportion for Sales

# Normality Test (needs nore than length of 3)
#stats.shapiro(sales) # Shapiro Test
#ShapiroResult(statistic=0.9649458527565002, pvalue=0.3199819028377533) >0.05 Accept Null hypothesis




# 2 Sample T test
scipy.stats.ttest_ind(sales.East, sales.West)
#Ttest_indResult(statistic=-0.82306722896822, pvalue=0.4969913379002262)
help(scipy.stats.ttest_ind)

scipy.stats.ttest_ind(sales.West, sales.North)
#Ttest_indResult(statistic=0.09642371171406917, pvalue=0.9319760699170774)

scipy.stats.ttest_ind(sales.North, sales.South)
#Ttest_indResult(statistic=0.4760613442985683, pvalue=0.6809649039074765)

#  Conclusion::No difference in  Proportion of sales in any Direction by Men and Women





## Using One way Anova
# One - Way Anova
F, p = stats.f_oneway(sales.East, sales.West, sales.South, sales.North)
F #0.3068565327605033
p #0.8204060154887464 Accept null Hypothesis
