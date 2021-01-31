library(readr)

cal <- read.csv(file.choose())

# Exploratory data analysis
summary(cal)
describe(cal)

library(Hmisc)
??Hmisc

is.null(cal)
sum(is.na(cal))

library("lattice") # dotplot is part of lattice package

#  Dotplot                               Graphical exploration
dotplot(cal$Weight.gained..grams., main = "Dot Plot of Weigth Gained (Grams)")
dotplot(cal$Calories.Consumed, main = "Dot Plot of Calories consumed")

# boxplot
boxplot(cal$Weight.gained..grams., col = "dodgerblue4")
boxplot(cal$Calories.Consumed, col = "red", horizontal = T)

# histogram
hist(cal$Weight.gained..grams.)
hist(cal$Calories.Consumed)


# Normal QQ plot   (Regression Line) for weight gained
qqnorm(cal$Weight.gained..grams.)
qqline(cal$Weight.gained..grams.)

# Normal QQ plot for Cal consumed
qqnorm(cal$Calories.Consumed)
qqline(cal$Calories.Consumed)

# hist for Cal consumed   (hist with kde)
hist(cal$Calories.Consumed, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(cal$Calories.Consumed))             # add a density estimate with defaults
lines(density(cal$Calories.Consumed, adjust = 2), lty = "dotted")   # add another "smoother" density

# Bivariate analysis
# Scatter plot
plot(cal$Calories.Consumed, cal$Weight.gained..grams., main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "calories consumed", 
     ylab = "Weight gained", pch = 20)  # plot(x,y)


## Checking its correlation

attach(cal)

# Correlation Coefficient   -----0.946991
cor(Weight.gained..grams., Calories.Consumed)

# Covariance    ------ 237669.5 (covariance refers to the measure of the directional relationship between two random variables)
cov(Weight.gained..grams., Calories.Consumed)


#################################### Linear Regression model #################################
reg <- lm(Weight.gained..grams. ~ Calories.Consumed, data = cal) # Y ~ X
?lm
summary(reg)

confint(reg, level = 0.95)
?confint

pred <- predict(reg, interval = "predict")  # predict(model) no test data yet. direct model
pred <- as.data.frame(pred)

View(pred) # pred has cols -> fit lower upper thats why pred$fit
?predict

# ggplot for adding Regression line for data
library(ggplot2)

ggplot(data = cal, aes(Calories.Consumed, Weight.gained..grams.) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ x)

# Alternate way
ggplot(data = cal, aes(x = Calories.Consumed, y = Weight.gained..grams.)) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = wc.at, aes(x = Calories.Consumed, y = pred$fit))

# Evaluation the model for fitness   0.946991 
cor(pred$fit, cal$Weight.gained..grams.) 

reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse
 ## RMSE for LM model simple --> 103.3025

#######################################################################################
##################################### Transformation Techniques

# 1. LOG ()
# input = log(x); output = y

plot(log(cal$Calories.Consumed), cal$Weight.gained..grams.)
cor(log(cal$Calories.Consumed), cal$Weight.gained..grams.)


## Model building for LOG( )  transformation technique
reg_log <- lm(cal$Weight.gained..grams. ~ log(cal$Calories.Consumed), data = cal)
summary(reg_log)


?confint  # Confidence Intervals for Model Parameters
confint(reg_log,level = 0.95)

# pred
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)

cor(pred$fit, cal$Weight.gained..grams.)  # corr btw predicted'Weigth and actual weigth

rmse <- sqrt(mean(reg_log$residuals^2))
rmse
 ### 141.0054 ---> RMSE for LM in LOG() technique


# Regression line for data
ggplot(data = cal, aes(log(cal$Calories.Consumed), cal$Weight.gained..grams.) ) +
  geom_point(color = 'red') + stat_smooth(method = lm, formula = y ~ log(x))

###############################################################################################
# 2. EXPO()
# Log transformation applied on 'y'  ### Exponential technique  .
# input = x; output = log(y)

plot(cal$Calories.Consumed, log(cal$Weight.gained..grams.))
cor(Calories.Consumed, log(Weight.gained..grams.))

model_expo <- lm(log(Weight.gained..grams.) ~ Calories.Consumed, data = cal)
summary(model_expo)

predictions <- predict(model_expo, interval = "predict")
predictions <- as.data.frame(predictions)

# Direct model in  predict  because there is column 'residuals'  reg_log1$residuals

model_expo$residuals
sqrt(mean(model_expo$residuals^2)) 
##**************** 0.3068228 for Residuals from model

pred_exp <- exp(predictions)  # Antilog = Exponential function
pred_exp <- as.data.frame(pred_exp)
cor(pred_exp$fit, wc.at$AT)

res_log1 = cal$Weight.gained..grams. - pred_exp$fit
rmse <- sqrt(mean(res_log1^2))
rmse 
##****************** 118.0452  for Residual from  actual-prediction$fit

# Regression line for data
ggplot(data = cal, aes(Calories.Consumed, log(Weight.gained..grams.)) ) +
  geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)



####################################### Non-linear models = Polynomial models
#3.POLY()
#

model_reg2 <- lm(log(Weight.gained..grams.) ~ Calories.Consumed 
           + I(Calories.Consumed*Calories.Consumed), data = cal)
summary(model_reg2)

predlog <- predict(model_reg2, interval = "predict")
pred <- exp(predlog)

pred <- as.data.frame(pred)
cor(pred$fit, cal$Weight.gained..grams.)

res2 = Weight.gained..grams. - pred$fit
rmse <- sqrt(mean(res2^2))
rmse
####  117.4145 RMSE for actual - predictions$fit

# Regression line for data
ggplot(data = cal, aes(Calories.Consumed, log(Weight.gained..grams.)) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))



# Data Partition

library(caTools)
set.seed(0)
split <- sample.split(cal$Weight.gained..grams., SplitRatio = 0.8)# --------- both are X_train , X_test
train <- subset(cal, split == TRUE)
test <- subset(cal, split == FALSE)

# Polynomial gives the lowest RMSE  that will be the  used model with 117

### Final Model Building  using plynomial

model_poly <- lm(log(Weight.gained..grams.) ~ Calories.Consumed 
                 + I(Calories.Consumed*Calories.Consumed), data = train)
summary(model_poly)

confint(model_poly,level=0.95)

################################## predict on TEST Data
predictions_test_log <- predict(model_poly,interval = "confidence", newdata = test) # predict on Xtest and model

predict_original <- exp(predictions_test_log) 
# converting log values to original values because model was built like that

predict_original <- as.data.frame(predict_original)

##Actual  - predicted$fit for TESTdata
test_residual <- test$Weight.gained..grams. - predict_original$fit # calculate error/residual
test_residual

# error is Residual for test  ---> 63.05934
test_rmse <- sqrt(mean(test_residual^2))
test_rmse

###################################### predict on TRAIN data
predictions_train_log <- predict(model_poly, interval = "confidence", newdata = train)

predict_original_train <- exp(predictions_train_log) # converting log values to original values

predict_original_train <- as.data.frame(predict_original_train)

#Actual  - predicted$fit for TRAINdata
train_error <- train$Weight.gained..grams. - predict_original_train$fit # calculate error/residual
train_error

train_rmse <- sqrt(mean(train_error^2))
train_rmse
## RMSE for train error is  84.47751

# RMSE for train ==63.05934
#RMSE for test ==  84.47751



#### Algorithm
'first plot( x) plot(y) and log if its logar , expo ,poly 
then cor()

1. then model = lm( y ~ x ,data=cal)

predictions = predict( model) and make predictions as dataframe(predictions)

get RMSE rmse <- sqrt(mean(model$residuals^2)) # model has residual column

then ggplot(data,aes(x,y) ) +geompoint()+stat_smooth( method=lm, formula)


2.IN LOG WAY :#  input = log(x); output = y


3.IN EXPO WAY ::# input = x; output = log(y)

predictions = predict(model)

as.dataframe(predictions)

 Then get exp(predictions) and make it as.dataframe 
                exp_pred = exp(predictions)
                
***Get RMSE for residuals  through  predicitons
 
 Actual - predictions$fit will give residuals .get RMSE for residuals 
 (predictions cols are fit , lower ,upper)

res_log1 = y - predictions$fit
rmse <- sqrt(mean(res_log1^2))
rmse
                


 NOte: predictions has cols ( fit ,lower, upper)  RMSE for prediction 
 NOte: model has residuals                        RMSE for residuals


 ***Get RMSE for residuals  through model
model$residuals
rmse <- sqrt(mean(model$residuals^2))
rmse



3. IN POLY()  input = x & x^2 (2-degree) and output = log(y)


 First model = lm( log(y) ~ X+ I(X*X),data=cal)

 SEcond prediction = predict(model)
  Take expo of it
          prediction <- exp(prediction)
  Make it into Dataframe
          prediction <- as.data.frame(prediction)
          
          
 #################cor(prediction$fit, cal$Weight.gained..grams.)
 
 Get RMSE through predictions$fit

res2 = Weight.gained..grams. - predictions$fit
rmse <- sqrt(mean(res2^2))
rmse
'





