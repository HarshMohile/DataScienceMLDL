
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter # ,OrderedDict
import matplotlib.pyplot as plt
import numpy as np

phonedata= pd.read_csv("D:\\360Assignments\\Submission\\Association Rules 6\\myphonedata.csv")

phonedata1= phonedata.iloc[:,4:]

phonedata_apriori = apriori(phonedata1, min_support = 0.089, max_len = 4, use_colnames = True)
phonedata_apriori.sort_values('support',ascending = False)
#white phone bookdata has most frequency as suport value of 60% followed by blue

phonedata_apriori = apriori(phonedata1, min_support = 0.004, max_len = 2, use_colnames = True)
phonedata_apriori.sort_values('support',ascending = False)

prules = association_rules(phonedata_apriori, metric = "lift", min_threshold = 1)
#antecedant orange and consequent of white with lift and confidence both above 1 .they are closely related to each other

support=prules['support']
confidence=prules['confidence']

plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


