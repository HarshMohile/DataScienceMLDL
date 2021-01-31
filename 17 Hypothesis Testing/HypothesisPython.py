import pandas as pd
import scipy
from scipy import stats

############ 2 sample T Test ##################

# Load the data
prom = pd.read_excel("C:/Data/Promotion.xlsx")
prom

prom.columns = "InterestRateWaiver", "StandardPromotion"

# Normality Test
stats.shapiro(prom.InterestRateWaiver) # Shapiro Test

print(stats.shapiro(prom.StandardPromotion))
help(stats.shapiro)

# Variance test
scipy.stats.levene(prom.InterestRateWaiver, prom.StandardPromotion)
help(scipy.stats.levene)
# p-value = 0.287 > 0.05 so p high null fly => Equal variances

# 2 Sample T test
scipy.stats.ttest_ind(prom.InterestRateWaiver, prom.StandardPromotion)
help(scipy.stats.ttest_ind)


############# One - Way Anova ################

cof = pd.read_excel("C:/Data/ContractRenewal_Data(unstacked).xlsx")
cof
cof.columns = "SupplierA", "SupplierB", "SupplierC"

# Normality Test
stats.shapiro(cof.SupplierA) # Shapiro Test
stats.shapiro(cof.SupplierB) # Shapiro Test
stats.shapiro(cof.SupplierC) # Shapiro Test

# Variance test
help(scipy.stats.levene)
# All 3 suppliers are being checked for variances
scipy.stats.levene(cof.SupplierA, cof.SupplierB, cof.SupplierC)

# One - Way Anova
F, p = stats.f_oneway(cof.SupplierA, cof.SupplierB, cof.SupplierC)

# p value
p  # P High Null Fly
# All the 3 suppliers have equal mean transaction time



######### 2-proportion test ###########
import numpy as np

two_prop_test = pd.read_excel("D:\\360Assignments\\Submission\\17 Hypothesis Testing\\JohnyTalkers.xlsx")

from statsmodels.stats.proportion import proportions_ztest

tab1 = two_prop_test.Person.value_counts()
tab1
tab2 = two_prop_test.Drinks.value_counts()
tab2

# crosstable table
pd.crosstab(two_prop_test.Person, two_prop_test.Drinks)

count = np.array([58, 152])
nobs = np.array([480, 740])

stats, pval = proportions_ztest(count, nobs, alternative = 'two-sided') 
print(pval) # Pvalue 0.000

stats, pval = proportions_ztest(count, nobs, alternative = 'larger')
print(pval)  # Pvalue 0.999  

################ Chi-Square Test ################

Bahaman = pd.read_excel("C:/Data/Bahaman.xlsx")
Bahaman

count = pd.crosstab(Bahaman["Defective"], Bahaman["Country"])
count
Chisquares_results = scipy.stats.chi2_contingency(count)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square
