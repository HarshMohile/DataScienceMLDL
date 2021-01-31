
# ANN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

strt = pd.read_csv("D:\\360Assignments\\Submission\\25 NeuralNets ANN\\ANN Assignment\\50_Startups.csv")

strt.head()

strt.head()
strt.describe()

## isnull() comes from pd and  np,
strt.isnull().values.any()
strt.isnull().sum()

strt.columns

## New  york has the highest plot of all countries in terms of Marketing and RnD
# R&D Spend
plt.bar(height = strt['R&D Spend'],x=strt['State'])
plt.hist(strt['R&D Spend']) #histogram
plt.boxplot(strt['R&D Spend']) #boxplot

# marketing Spend
plt.bar(height = strt['Marketing Spend'], x=strt['State'])
plt.hist(strt['Marketing Spend']) #histogram
plt.boxplot(strt['Marketing Spend']) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=strt['R&D Spend'], y=strt['State'])
# No correlation between Rnd Spend  for each state


# Correlation matrix 
a = strt.corr()
a

#Rename a col with whitespace
strt.rename(columns={"Marketing Spend":"MarketingSpend"},inplace=True)
strt.rename(columns={"R&D Spend":"RnDSpend"},inplace=True)


# Replace 0 with mean 
strt = strt.mask(strt==0).fillna(strt.mean())

## State for State Column
# creating instance of labelencoder
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

# Assigning numerical values and storing in another column
strt['States_cat'] =  lb.fit_transform(strt['State'])

# Now drop State 
strt = strt.drop('State',axis=1)


# train test split 
from sklearn.model_selection import train_test_split

X= strt.drop('Profit',axis=1).values
y= strt['Profit'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

X_train.shape #4
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
    model.add(Dense(4,activation="relu"))
    model.add(Dense(4,activation="relu"))
    model.add(Dense(4,activation="relu"))
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
#[14042229760.0, 0.0]

# Further Eval on test data
test_score = model.evaluate(X_test,y_test,verbose=0)
#[14363037696.0, 0.0]

from sklearn.metrics import mean_squared_error, mean_absolute_error

predictions = model.predict(X_test)



mean_squared_error(y_test,predictions)
# 14363409193.50244

#RMSE
np.sqrt(mean_squared_error(y_test,predictions))
#119847.44133064519

mean_absolute_error(y_test,predictions)
#113826.2502284318

strt['Profit'].describe()
#mean  is 112012.639200 almost closer to the MEAN ABS ERROR 

plt.scatter(y_test,predictions)
