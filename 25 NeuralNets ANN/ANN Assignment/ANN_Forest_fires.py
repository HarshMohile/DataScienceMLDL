# ANN Forest Fires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

forest = pd.read_csv("D:\\360Assignments\\Submission\\25 NeuralNets ANN\\ANN Assignment\\fireforests.csv")
forest.describe()

## isnull() comes from pd and  np,
forest.isnull().values.any()
forest.isnull().sum()

forest.columns
    
# remove the first 2 columns as dummy variables are already present
forest.drop(['month','day'],inplace=True,axis=1)

#EDA 
sns.pairplot(forest, hue="size_category", palette="Dark2")

sns.heatmap(forest.corr(), annot=True)

sns.kdeplot( forest['temp'], forest['area'],
                 cmap="plasma", shade=True, shade_lowest=False)


# Form a facetgrid using columns with a hue 
graph = sns.FacetGrid(forest, col ="rain",  hue ="size_category") 
# map the above form facetgrid with some attributes 
graph.map(plt.scatter, "temp", "monthnov", edgecolor ="w").add_legend()


# train test split 
from sklearn.model_selection import train_test_split

X= forest.drop('area',axis=1).values
y= forest['area'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

X_train.shape #27
X_test.shape

#Scale the data using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Model 
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(27,activation="relu"))
    model.add(Dense(27,activation="relu"))
    model.add(Dense(27,activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mse",optimizer="adam",metrics=["accuracy"])
    return model

# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
model.fit(x=X_train,y=y_train,epochs=20)

model.history.history


# Plotting the loss
loss = model.history.history['loss']

sns.lineplot(x=range(len(loss)),y=loss)
plt.title("Training Loss per Epoch");

#metrices Used
model.metrics_names

# Further Eval on train data
training_score = model.evaluate(X_train,y_train,verbose=0)
# [2325.1728515625, 0.0]

# Further Eval on test data
test_score = model.evaluate(X_test,y_test,verbose=0)
#[7941.91650390625, 0.0]

from sklearn.metrics import mean_squared_error, mean_absolute_error

predictions = model.predict(X_test)



mean_squared_error(y_test,predictions)
# 7941.9169801851764

#RMSE
np.sqrt(mean_squared_error(y_test,predictions))
#89.11743364900705

mean_absolute_error(y_test,predictions)
#20.253505704464057

forest['area'].describe()
#mean  is 12.847292 almost closer to the MEAN ABS ERROR  20.253505704464057

plt.scatter(y_test,predictions)






