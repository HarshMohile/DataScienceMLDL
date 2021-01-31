import pandas as pd
import numpy as np
import scipy
from scipy import stats 
import statsmodels.api as sm

lab= pd.read_csv("D:\\360Assignments\\Submission\\17 Hypothesis Testing\\lab_tat_updated.csv")

# Check whether the TAT of labs matter if they are different avg TAT
# Ho --> No difference in TurnAround Time for labs
# H1 --> There is significant difference TAT for labs

## More than 2 number  of numerical columns 
lab.columns
# Normality Test
stats.shapiro(lab['Laboratory_1']) # Shapiro Test
#ShapiroResult(statistic=0.9886691570281982, pvalue=0.42317795753479004)

stats.shapiro(lab['Laboratory_2']) # Shapiro Test
#ShapiroResult(statistic=0.9936322569847107, pvalue=0.8637524843215942)

stats.shapiro(lab['Laboratory_3']) # Shapiro Test
#ShapiroResult(statistic=0.9796067476272583, pvalue=0.06547004729509354)

#Through shapiro  Test we accept null hypothesis that there is No difference in TurnAround Time for labs

# Variance test
help(scipy.stats.levene)
# All 3 suppliers are being checked for variances
scipy.stats.levene(lab['Laboratory_1'], lab['Laboratory_2'], lab['Laboratory_3'],lab['Laboratory_4'])
#LeveneResult(statistic=1.025294593220823, pvalue=0.38107781677304564)


# One - Way Anova
F, p = stats.f_oneway(lab['Laboratory_1'], lab['Laboratory_2'], lab['Laboratory_3'],lab['Laboratory_4'])
p

# Chi Square test
count1 = pd.crosstab(lab['Laboratory_1'], lab['Laboratory_2'])
count1
Chisquares_results = scipy.stats.chi2_contingency(count1)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square
# [['Test Statistic', 'p-value'], [13720.000000000005, 0.18108894174923318]]
#High p value than 0.05
count2 = pd.crosstab(lab['Laboratory_2'], lab['Laboratory_3'])
count2
Chisquares_results = scipy.stats.chi2_contingency(count2)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square