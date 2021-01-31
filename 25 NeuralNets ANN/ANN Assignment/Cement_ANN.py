# ANN Cement
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

cem = pd.read_csv("D:\\360Assignments\\Submission\\25 NeuralNets ANN\\ANN Assignment\\concrete.csv")
cem.describe()

## isnull() comes from pd and  np,
cem.isnull().values.any()
cem.isnull().sum()

# Data contains lots of 0 . we can fill wiith using One Hot Encoding by flling it with mean
cem = cem.mask(cem==0).fillna(cem.mean())

# Data visualization
sns.heatmap(cem.corr(),annot= True)

sns.kdeplot( cem['age'], cem['strength'],
                 cmap="plasma", shade=True, shade_lowest=False)


# train test split 
from sklearn.model_selection import train_test_split

X= cem.drop('strength',axis=1).values
y= cem['strength'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

X_train.shape #8
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
    model.add(Dense(8,activation="relu"))
    model.add(Dense(8,activation="relu"))
    model.add(Dense(8,activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mse",optimizer="adam",metrics=["accuracy"])
    return model

# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
model.fit(x=X_train,y=y_train,epochs=60)

model.history.history


# Plotting the loss
loss = model.history.history['loss']

sns.lineplot(x=range(len(loss)),y=loss)
plt.title("Training Loss per Epoch");

#metrices Used
model.metrics_names

# Further Eval on train data
training_score = model.evaluate(X_train,y_train,verbose=0)
# [143.2986602783203, 0.0]

# Further Eval on test data
test_score = model.evaluate(X_test,y_test,verbose=0)
#[143.02023315429688, 0.0]

from sklearn.metrics import mean_squared_error, mean_absolute_error

predictions = model.predict(X_test)


mean_squared_error(y_test,predictions)
# 143.02024102256843

#RMSE
np.sqrt(mean_squared_error(y_test,predictions))
#11.95910703282517

mean_absolute_error(y_test,predictions)
#9.790517534175738

cem['strength'].describe()
#mean  is 35.817961 almost closer to the MEAN ABS ERROR  9.790517534175738

plt.scatter(y_test,predictions)












