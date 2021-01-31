import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


air = pd.read_excel("D:\\360Assignments\\Submission\\24 Time Series Forecasting\\Forcasting TS Assignment\\Airlines Data.xlsx")

air.Passengers.plot() # time series plot

## isnull() comes from pd and  np,
air.isnull().values.any()
air.isnull().sum()

air.columns

air.info()

# Converting the Month to Datatime format
#air['Month'] = pd.to_datetime(air['Month'])

# removing the  HH:MM:ss part
air['Month'] = air['Month'].dt.date

air.set_index('Month',inplace=True)

# Test for Stationarity by ADFuller Test by hypotheisis testing
from statsmodels.tsa.stattools import adfuller

adf = adfuller(air['Passengers'])

#Ho - it is Non stationary
# H1 - It is stationary

# Through Method 
def adfuller_test(passengers):
    result =adfuller(passengers)
    labels=['ADF Test Statistics','p-value','#Lags USed','Number of Observation USed']
    
    for results,labels in zip(result,labels):
        print(labels+ " : " + str(results))
        
    if result[1] >= 0.05:
       print("Accept Null hypothesis. It is not Stationary.")
    else:
        print("Reject NullHypothesis . It is Stationary")


adfuller_test(air['Passengers'])
# p -value is 0.9 .So data is Stationary.
 

# Differncing  to remove Stationarity( to 12 month becuse in plot the seasonal is after every 12 months)
air['Seasonal first differnce '] = air["Passengers"] -air["Passengers"].shift(12) 

# applying dicky duller on transfomred data 
    
adfuller_test(air['Seasonal first differnce '].dropna())

# p-value : 0.07578397625851772 .So here we reject Null Hypothesis and say that data is  stationary

# plot just to check 
air['Seasonal first differnce '].plot()

############## Plotting using ACF ,PACF( another package)

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

fig=plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(air['Seasonal first differnce '].iloc[13:], lags=40, ax=ax1)  
## Axes to plot 2 graphs below one and another
ax2 = fig.add_subplot(212)
fig = plot_pacf(air['Seasonal first differnce '].iloc[13:], lags=40, ax=ax2)  

# ##############################  ARIMA FOR SEASONAL DATA 
# for Ar the shutssoff happens at 1 and fro MA the exponential deccrease witihin the blue part is still 3 
# (p =1 , q =3 ) 
from statsmodels.tsa.arima_model import ARIMA

model=ARIMA(air['Passengers'],order=(1,1,3))
model_fit=model.fit()


#Predict
air['forecast']=model_fit.predict(start=80,end=100,dynamic=True)
# PLot
air[['Passengers','forecast']].plot(figsize=(12,8))


#########################      SARIMA SEASONAL ARIMA

import statsmodels.api as sm

model_sarima=sm.tsa.statespace.SARIMAX(air['Passengers'],order=(1, 1, 3),seasonal_order=(1,1,3,12))
model_sarima_fit =model_sarima.fit(maxiter=200, method='nm')

#Predict
air['forecast']=model_sarima_fit.predict(start=80,end=100,dynamic=True)
#Plot
air[['Passengers','forecast']].plot(figsize=(12,8))
 


# #################### Creating a dummy test dataframe for future predictions

from pandas.tseries.offsets import DateOffset

# air.index (random date from your air dataset from last  index)
# Predict for the next 2 yr(24) in x=24 and months=24

future_dates=[air.index[-5]+ DateOffset(months=x)for x in range(0,24)]

future_datest_df = pd.DataFrame(index = future_dates[1:],columns=air.columns)


future_df=pd.concat([air,future_datest_df])

########################### predict  using SARIMA model
future_df['forecast'] = model_sarima_fit.predict(start = 80, end = 120, dynamic= True)  
future_df[['Passengers', 'forecast']].plot(figsize=(12, 8))


# sart =80 for both during SARMIA initially to predict and then start =80 for when we create new data and predict future Passengers