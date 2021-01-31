
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter # ,OrderedDict
import matplotlib.pyplot as plt

groceries = []
with open("D:\\360Assignments\\Submission\\Association Rules 6\\groceries.csv") as f:
    groceries = f.read()
    
groceries = groceries.split("\n")

#Seperated each 
groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))

# every item is a different row now to get its frequency
all_groceries_list = [i for item in groceries_list for i in item]

#get every item frequencies 
item_frequencies = Counter(all_groceries_list)

item_frequencies #pork': 567, item: freq

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

item_frequencies # here they are ASC order
# Storing frequencies and items in separate variables  in Desc orer .highest frequency first
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))


# most frequency support item barplot
plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = 'rgbkymc')
plt.xticks(list(range(0, 11), ), items[0:11])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()

#"whole milk" with other "vegetables" are nost frequently occured with high support value
groceries_list
#Create a DataFrame for apriori/assoc algorithm
groceries_df = pd.DataFrame(pd.Series(groceries_list))
groceries_df = groceries_df.iloc[:9835, :] # column name is 0
groceries_df.columns = ["transactions"] # column name given :transactions

X = groceries_df['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

# min_support given as 0.004 :
groceries_apriori = apriori(X, min_support = 0.0040, max_len = 2, use_colnames = True)

# min_support given as 0.0067 : highest --> whole milk and other vegetables
groceries_apriori = apriori(X, min_support = 0.0067, max_len = 2, use_colnames = True)

groceries_apriori 

#view based on lift metric. frequently brought together 
rules = association_rules(groceries_apriori, metric = "lift", min_threshold = 1)
#highest leverage that occurs in the dataset
rules.sort_values('lift', ascending = False).head(10)

 #             antecedents           consequents  ...  leverage  conviction
#450     (root vegetables)               (herbs)  ...  0.005243    1.051406
