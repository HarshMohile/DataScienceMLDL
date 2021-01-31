import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter # ,OrderedDict
import matplotlib.pyplot as plt
import numpy as np

retail=[]
with open("D:\\360Assignments\\Submission\\Association Rules 6\\transactions_retail1.csv") as f:
    retail = f.read()

retail = retail.split("\n")

retail_list = []
for i in retail:
    retail_list.append(i.split(","))


all_retail_list = [i for item in retail_list for i in item]

item_frequencies = Counter(all_retail_list)

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))


plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = 'rgbkymc')
plt.xticks(list(range(0, 11), ), items[0:11])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()

retail_df = pd.DataFrame(pd.Series(retail_list))


retail_df.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = retail_df['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.4, max_len = 2, use_colnames = True)
frequent_itemsets
frequent_itemsets = apriori(X, min_support = 0.006, max_len = 2, use_colnames = True)
frequent_itemsets

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)


