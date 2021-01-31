import pandas as pd
import numpy as np
import scipy
from scipy import stats 
import statsmodels.api as sm


cust= pd.read_csv("D:\\360Assignments\\Submission\\17 Hypothesis Testing\\CustomerOrderform.csv")

'''
lab_tat_updated
CustomerOrderform
Fantaloons
BuyerRatio
Cutlets
'''

cust.columns

# Check whether the 
# Ho --> No error /defect in Customer form
# H1 --> There is significant  error in Customer Form

#The c2 test is used to determine whether an association
# (or relationship) between 2 categorical variables in a sample is likely 
#to reflect a real association between these 2 variables in the population.



# Chi Square test
count1 = pd.crosstab(cust['Phillippines'], cust['Indonesia'])
count1
Chisquares_results = scipy.stats.chi2_contingency(count1)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square
 #[['Test Statistic', 'p-value'], [0.1856391005881107, 0.6665712150680798]]
#High p value than 0.05

count2 = pd.crosstab(cust['Malta'], cust['India'])
count2
Chisquares_results = scipy.stats.chi2_contingency(count2)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square
#[['Test Statistic', 'p-value'], [1.419106436194816, 0.23355053527979247]]

#we accept null hypothesis which there is no error in customer form or defect

 