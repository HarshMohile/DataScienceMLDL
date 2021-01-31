
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter # ,OrderedDict
import matplotlib.pyplot as plt

books= pd.read_csv("D:\\360Assignments\\Submission\\Association Rules 6\\book.csv")

books_apriori =apriori(books,min_support = 0.05, max_len = 4, use_colnames = True)
# Most Frequent item sets based on support 
books_apriori.sort_values('support', ascending = False, inplace = True)
books_apriori

#highest spport : childbooks adn cookbooks (having most frequency)

books_apriori =apriori(books,min_support = 0.6, max_len = 4, use_colnames = True)
books_apriori

plt.bar(x = list(range(0, 11)), height = books_apriori.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), books_apriori.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()


brules = association_rules(books_apriori, metric = "lift", min_threshold = 1)
brules.head(20)
brules.sort_values('lift', ascending = False)
# antecedents (youthbooksmcookbooks) consequent(italCook) highest lift =
type(brules)
brules.unique()
brules_unique=brules.antecedents.unique
plt.scatter(x=brules["support"], y=brules["confidence"])
