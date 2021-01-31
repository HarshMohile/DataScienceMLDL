library(readxl)
air <- read_excel(file.choose()) 
View(air) # Seasonality 12 months


# Creating  table for t ,t2 , log(Passengers) in dataframe
# Pre Processing
# input t
air["t"] <- c(1:96)
View(air)

air["t_square"] <- air["t"] * air["t"]
air["log_Passengers"] <- log(air["Passengers"])


#air <- air[,c(-3,-4,-5)]

####----- -----------------------Splitting the dataset -------------------

library(caTools)
set.seed(0)
split <- sample.split(air$Passengers)# ---------
train <- subset(air, split == TRUE)
test <- subset(air, split == FALSE)


########################### LINEAR MODEL #############################
attach(air)

linear_model <- lm(Passengers ~ t, data = train)
summary(linear_model)
#Predict
linear_pred <- data.frame(predict(linear_model, interval = 'predict', newdata = test))
#RMSE
rmse_linear <- sqrt(mean((test$Passengers - linear_pred$fit)^2, na.rm = T))
rmse_linear
#32.28634
######################### Exponential ############################

expo_model <- lm(log_Passengers ~ t, data = train)
summary(expo_model)
expo_pred <- data.frame(predict(expo_model, interval = 'predict', newdata = test))

rmse_expo <- sqrt(mean((test$Passengers - exp(expo_pred$fit))^2, na.rm = T))
rmse_expo
#33.29019

######################### Quadratic ###############################
#y +t+ t2

Quad_model <- lm(Passengers ~ t + t_square, data = train)
summary(Quad_model)
Quad_pred <- data.frame(predict(Quad_model, interval = 'predict', newdata = test))

rmse_Quad <- sqrt(mean((test$Passengers - Quad_pred$fit)^2, na.rm = T))
rmse_Quad

#34.0609

####################### Predicting new data (Linear had lowest RMSE )#############################
test_data <- air[5,]

View(test_data)
pred_new <- predict(linear_model, newdata = test_data, interval = 'predict')
pred_new <- as.data.frame(pred_new)
pred_new$fit
plot(linear_model)

############  ACF for getting Q value (past error ) Moving Averages .PACF for p value

# ARima ( data is model$rediuals)


# Differences 

library(forecast)
y <- ts(air$Passengers, frequency= 12)
y.diff <- diff(y ,lag=40 ,differnces =12)
ggtsdisplay(y.diff)

# Seperately using the model$ residuals
# ACF plot
acf(linear_model$residuals, lag.max = 40) #  q =1 take all residual value of the model built & plot ACF plot
#PACF 
pacf(linear_model$residuals, lag.max = 40) # p =1

install.packages("astsa")

library(astsa)


model <- arima(y.diff,order=c(1, 1, 1),seasonal = list(order=c(1,1,1),period=12),)
summary(model)
model$coef
model$residuals

SARerrors <- model$residuals

acf(SARerrors, lag.max = 12)

# predicting next 12 months errors using arima(order=c(1,0,0))

library(forecast)
errors_12 <- forecast(model, h = 12)

View(errors_12)

future_errors <- data.frame(errors_12)
class(future_errors)
fpe <- future_errors$Point.Forecast






