

#importing necessary libraries
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

#importing the dataset
Data_Cleaned = pd.read_csv('D:\\Wplace\\WorkFiles\\360\\Association rule\\retaildata.csv', index_col = 'InvoiceDate')
Data_Cleaned.index = pd.to_datetime(Data_Cleaned.index, format = '%Y-%m-%d %H:%M') # box = False)

#converting the data into the standard form
Baskets = Data_Cleaned.loc[(Data_Cleaned['Quantity']>0) ,['InvoiceNo','Description','Quantity']]
Baskets = Baskets.groupby(['InvoiceNo','Description'])['Quantity'].sum().unstack(fill_value=0)
Baskets = (Baskets > 0)


#finding frequent itemsets and association rules
frequent_itemsets = apriori(Baskets, min_support=0.0325, use_colnames=True)
rules = association_rules(frequent_itemsets, metric = 'confidence', min_threshold=0.5)
rules