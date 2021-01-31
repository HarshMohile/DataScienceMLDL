library(forecast)
library(fpp)
library(smooth) # forsmoothing and MAPE
library(tseries)
library(readxl)

cc <- read_excel(file.choose())

colnames(cc)

# Converting Sales into Time Series object
tsSales <- ts(cc$Sales, frequency =4 )

####----- -----------------------Splitting the dataset -------------------

library(caTools)
set.seed(0)
split <- sample.split(cc$Sales)# ---------
train <- subset(tsSales, split == TRUE)
test <- subset(tsSales, split == FALSE)


# Considering only 4 Quarters of data for testing because data itself is Quarterly
# seasonal data

# converting time series object
train <- ts(train, frequency = 4)
test <- ts(test, frequency = 4)


# Plotting time series data
plot(tsSales)
# Visualization shows that it has level, trend, seasonality , noise at start=> Additive seasonality

######################################### MOVING AVERAGE ######

class(train)

ma_model1 <- sma(train, n=4)
?sma
ma_pred <- data.frame(predict(ma_model1, h = 4))
ma_pred

# PLotting the forecast model

plot(forecast(ma_model1))


# MAPE  test[1:4] because 4 Quarters

ma_mape <- MAPE(ma_pred$Point.Forecast, test[1:4])*100
ma_mape
#58.8312

################################### USING HoltWinters function ################
# Optimum values
# with alpha = 0.2 which is default value
# Assuming time series data has only level parameter

hw_a <- HoltWinters(train, alpha = 0.2, beta = F, gamma = F)
hw_a
hwa_pred <- data.frame(predict(hw_a, n.ahead = 4))

plot(forecast(hw_a, h = 4))

# MAPE

hwa_mape <- MAPE(hwa_pred$fit, test[1:4])*100   #52.17133

#### ################# HYPERPARAMETER TUNING WITH ALPHA BETA , N.AHEAD #########################
##########################################   ON HOLTS WINTER ############################################
# The data has trend , level
                            #(1)
hw_ab <- HoltWinters(train, alpha = 0.2, beta = 0.15, gamma = F)
hw_ab
hwab_pred <- data.frame(predict(hw_ab, n.ahead = 4))


# by looking at the plot the forecasted values are still missing some characters exhibited by traindata
#PLott
plot(forecast(hw_ab, h = 4))

# MAPE ab  
hwab_mape <- MAPE(hwab_pred$fit,test[1:4])*100 #57.51578


# Assuming time series data has level,trend and seasonality
                             #(2)

hw_abg <- HoltWinters(train, alpha = 0.8, beta = 0.16, gamma = 0.05)
hw_abg
hwabg_pred <- data.frame(predict(hw_abg, n.ahead = 4))


# by looking at the plot the characters of forecasted values are closely following historical data

plot(forecast(hw_abg, h = 4))
hwabg_mape <- MAPE(hwabg_pred$fit, test[1:4])*100

#60.15435

# Holst winter with alpha 0.2 and beta set to False has lowest MAPE 

df_mape <- data.frame(c("ma_mape","hwa_mape","hwabg_mape")
                      ,c(ma_mape,hwa_mape,hwabg_mape))

colnames(df_mape)<-c("MAPE","VALUES")
View(df_mape)

# Based on the MAPE value who choose holts winter exponential tecnique which assumes the time series
# Data level, trend, seasonality characters with default values of alpha, beta and gamma

new_model <- HoltWinters(tsSales)
new_model

plot(forecast(new_model, n.ahead = 4))

# Forecasted values for the next 4 quarters
forecast_new <- data.frame(predict(new_model, n.ahead = 4))
forecast_new
