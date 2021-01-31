import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 


cc = pd.read_excel("D:\\360Assignments\\Submission\\24 Time Series Forecasting\\Forcasting TS Assignment\\CocaCola_Sales_Rawdata.xlsx")

cc.Sales.plot() # time series plot 

cc.plot(x='Quarter' , y= 'Sales')
plt.show()

cc.describe()
# there is trend upward with bit if noise present in Cococola Dataset

    


month_Var = cc

# Fetch year
month_Var = cc['Quarter'].apply(lambda x : x.split('_')[1])


# moving Average     To Show Trend Analysis
cc.Sales.plot(label = "org")
for i in range(2, 9, 2):
    cc["Sales"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 9)
# Rolling Statistics: A common technique to assess the constancy of a model’s parameters is to compute parameter estimates over a rolling window of a fixed size through the sample. 

ccSales = cc[["Sales"]]

# Seaonability Analysis -1 
ccSales.diff(periods=4).plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.show()

# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(cc.Sales, model = "additive", period = 4)
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(cc.Sales, model = "multiplicative", period = 4)
decompose_ts_mul.plot()


# ACF PLot to get the (q) direct and indirect relataions /effect on 2 timestamps (MA Model)
import statsmodels.graphics.tsaplots as tsa_plots

tsa_plots.plot_acf(cc.Sales, lags = 4) ## all lags are above error band

# PACF PLot to get the (p) direct and avoid  indirect relataions /effect on 2 timestamps  (AR model )
tsa_plots.plot_pacf(cc.Sales, lags=4)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
Train, Test = train_test_split(cc, test_size = 0.2) # 20% test data

#reset index
Test.set_index(np.arange(1,10),inplace=True)
Train.set_index(np.arange(1,34),inplace=True)

 ############################# EXPONENTIAL SMOOTHING MODEL #########################
# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])



# MAPE
MAPE_Simple = np.mean(np.abs((pred_ses -Test.Sales)/ Test.Sales)*100)
#28.28529617659357


# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])


# MAPE
MAPE_Holt = np.mean(np.abs((pred_hw -Test.Sales)/ Test.Sales)*100)
#35.30989441123007


# Winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"],
                                         seasonal = "add",
                                         trend = "add", 
                                         seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])


MAPE_win1 = np.mean(np.abs((pred_hwe_add_add -Test.Sales)/ Test.Sales)*100)
#27.903811698151365 

# Winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],
                                         seasonal = "mul",
                                         trend = "add",
                                         seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])

MAPE_win2 = np.mean(np.abs((pred_hwe_mul_add -Test.Sales)/ Test.Sales)*100) 
#32.691647924705855

# Storing MAPE values in a data frame
dict_1 = {'MAPE':['MAPE_Simple', 'MAPE_Holt', 'MAPE_win1', 'MAPE_win2'], 
                'Values':[MAPE_Simple, MAPE_Holt, MAPE_win1,MAPE_win2]}

type(dict_1)
mape_df = pd.DataFrame(dict_1)  
mape_df



