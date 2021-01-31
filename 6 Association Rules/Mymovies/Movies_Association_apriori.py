
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter # ,OrderedDict
import matplotlib.pyplot as plt
import numpy as np



movies= pd.read_csv("D:\\360Assignments\\Submission\\Association Rules 6\\my_movies.csv")

movies= movies.iloc[:,5:]

movies_apriori = apriori(movies, min_support = 0.0040, max_len = 2, use_colnames = True)
movies_apriori.sort_values('support',ascending = False)


# with 0.7 support gladiator and Sixth sense(highest support)

movies_apriori = apriori(movies, min_support = 0.4, max_len = 4, use_colnames = True)
movies_apriori.sort_values('support',ascending = False)


#view based on lift metric. frequently brought together 
mrules = association_rules(movies_apriori, metric = "lift", min_threshold = 1)
#gladiator and sixth sense are the most recommended frequently watched together with lift od 1.19(highest)

support=mrules['support']
confidence=mrules['confidence']

for i in range (len(support)):
   support[i] = support[i] + 0.0025 * (np.random.randint(1,10) - 5) 
   confidence[i] = confidence[i] + 0.0025 * (np.random.randint(1,10) - 5)


plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()