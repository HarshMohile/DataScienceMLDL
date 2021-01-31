# CIFAR10 CNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

#pip install keras-tuner

from tensorflow.keras import datasets ,layers ,models
from tensorflow import keras

# Data loading
(X_train , y_train) ,(X_test, y_test) = datasets.cifar10.load_data() 

X_train.shape
# train sameple :50k , each sample is 32 *32 , 3 is RGB channel

X_test.shape

plt.imshow(X_train[0]) # view images 

# reshaping the dataset y_Train from 2d array to just normal array [Flatten]

y_train[:5]
y_train.shape
y_train = y_train.reshape(-1,)


# Scaling 0-255
X_train = X_train/255
X_test = X_test /255

#Sigmoid  gives probab as op  0.4 0.67  that added together doesnt equals 100 
# Softmax --- //  ---  0.45 0.59 such that  a+b =100  cuz a/a+b   b/a+b


## model 

def build_model(hp):  
  model = keras.Sequential([
    keras.layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
        activation='relu',
        input_shape=(32,32,3)
        
    ),
    keras.layers.MaxPooling2D((2,2)),
    
    keras.layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        activation='relu'
    ),
     keras.layers.MaxPooling2D((2,2)),
     
    keras.layers.Flatten(),
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ),
    keras.layers.Dense(10, activation='softmax')
  ])
  
  model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  
  return model
    
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters


tuner=RandomSearch(build_model,
                          objective='val_accuracy',
                          max_trials=5,directory='output',project_name="CIFAR10")


# model.fit(X_Train ,y_train)
tuner.search(X_train,y_train,epochs=3,validation_split=0.1)