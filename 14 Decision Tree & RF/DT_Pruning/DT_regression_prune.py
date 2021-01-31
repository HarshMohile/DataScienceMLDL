import pandas as pd


'''
Pruning
considers each of the decision nodes in the tree to be a candidate for pruning

Pruning a DN consists of removing the subtree rooted at that node.Making it a leaf node, and assigning
it the most common classification of the training examples affiliated with that node

Nodes are removed only if the resulting pruned tree performs no worse than the
original over the validation set

Pruning of nodes continues further pruning is harmful(i.e decreases accuracy of the tree over the 
                                                      validation set)

'''

df = pd.read_csv("C:/Datasets_BA/360DigiTMG/DS_India/360DigiTMG DS India Module wise PPTs/Module 14 Decision Tree/New code-pruning/DT_Regression/movies.csv")
df.info()


# Dummy variables
df.head()

# n-1 dummy variables will be created for n categories
df = pd.get_dummies(df, columns = ["3D_available", "Genre"], drop_first = True)

df.head()

# Input and Output Split
predictors = df.loc[:, df.columns!="Collection"]
type(predictors)

target = df["Collection"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)

# Train the Regression DT
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 3)
regtree.fit(x_train, y_train)

# Prediction
test_pred = regtree.predict(x_test)
train_pred = regtree.predict(x_train)

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score

# Error on test dataset
mean_squared_error(y_test, test_pred)
r2_score(y_test, test_pred)

# Error on train dataset
mean_squared_error(y_train, train_pred)
r2_score(y_train, train_pred)


# Plot the DT
#dot_data = tree.export_graphviz(regtree, out_file=None)
#from IPython.display import Image
#import pydotplus
#graph = pydotplus.graph_from_dot_data(dot_data)
#Image(graph.create_png())

# Pruning the Tree
# Minimum observations at the internal node approach
regtree2 = tree.DecisionTreeRegressor(min_samples_split = 3)
regtree2.fit(x_train, y_train)

# Prediction
test_pred2 = regtree2.predict(x_test)
train_pred2 = regtree2.predict(x_train)

# Error on test dataset
mean_squared_error(y_test, test_pred2)
r2_score(y_test, test_pred2)

# Error on train dataset
mean_squared_error(y_train, train_pred2)
r2_score(y_train, train_pred2)

###########
## Minimum observations at the leaf node approach
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf = 3)
regtree3.fit(x_train, y_train)

# Prediction
test_pred3 = regtree3.predict(x_test)
train_pred3 = regtree3.predict(x_train)

# measure of error on test dataset
mean_squared_error(y_test, test_pred3)
r2_score(y_test, test_pred3)

# measure of error on train dataset
mean_squared_error(y_train, train_pred3)
r2_score(y_train, train_pred3)
