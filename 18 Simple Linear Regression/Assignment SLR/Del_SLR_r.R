library(readr)

del <- read.csv(file.choose())

# Exploratory data analysis
summary(del)
describe(del)

library(Hmisc)
??Hmisc

is.null(del)
sum(is.na(del))

library("lattice") # dotplot is part of lattice package

#  Dotplot                               Graphical exploration
dotplot(del$Delivery.Time, main = "Dot Plot of Delivery Time")
dotplot(del$Sorting.Time, main = "Dot Plot of Sorting Time")

# boxplot
boxplot(del$Delivery.Time, col = "dodgerblue4")
boxplot(del$Sorting.Time, col = "red", horizontal = T)

# histogram
hist(del$Delivery.Time)
hist(del$Sorting.Time)


# Normal QQ plot   (Regression Line) for weight gained
qqnorm(del$Delivery.Time)
qqline(del$Delivery.Time)

# Normal QQ plot for Cal consumed
qqnorm(del$Sorting.Time)
qqline(del$Sorting.Time)


# Bivariate analysis
# Scatter plot
plot(del$Sorting.Time, del$Delivery.Time, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "calories consumed", 
     ylab = "Weight gained", pch = 20)  # plot(x,y)
## slight regression line forming a correlation line 


## 0.83 correaltion formed with these 2 cols 
library(corrplot)
cor <- cor(del)
corrplot(cor,method="number")

#################################### Linear Regression model #################################
model <- lm(Delivery.Time ~ Sorting.Time, data = del) # Y ~ X
model

model$residuals
residuals_rmse <- sqrt(mean(model$residuals^2))  
residuals_rmse

####  RMSE recorded as 2.79165


pred <- predict(model,interval = "predict")
pred <- as.data.frame(pred)

# ggplot for adding Regression line for data
library(ggplot2)

ggplot(data = del, aes(Sorting.Time, Delivery.Time) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ x)

cor(pred$fit, del$Delivery.Time) 
# 0.8259973 for predicted value  to the actual observed value

#######################################################################################
##################################### Transformation Techniques

# 1. LOG ()
# input = log(x); output = y

plot(log(del$Sorting.Time), del$Sorting.Time)
cor(log(del$Sorting.Time), del$Sorting.Time)


# Model building for LOG( )  transformation technique
reg_log <- lm(del$Delivery.Time ~ log(del$Sorting.Time), data = del)
summary(reg_log)


?confint  # Confidence Intervals for Model Parameters
confint(reg_log,level = 0.95)

# pred
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)

cor(pred$fit, del$Delivery.Time)  # corr btw predicted'del time and actual Del time is 0.83

rmse <- sqrt(mean(reg_log$residuals^2))
rmse
###  2.733171 ---> RMSE for LM in LOG() technique


# Regression line for data
ggplot(data = cal, aes(log(cal$Calories.Consumed), cal$Weight.gained..grams.) ) +
  geom_point(color = 'red') + stat_smooth(method = lm, formula = y ~ log(x))


####################################### Non-linear models = Polynomial models
#3.POLY()
#

model_reg2 <- lm(log(del$Delivery.Time) ~ del$Sorting.Time 
                 + I(del$Sorting.Time*del$Sorting.Time), data = del)
summary(model_reg2)

predlog <- predict(model_reg2, interval = "predict")
pred <- exp(predlog)

pred <- as.data.frame(pred)
cor(pred$fit, del$Delivery.Time)

res2 = del$Delivery.Time - pred$fit
rmse <- sqrt(mean(res2^2))
rmse

model_reg2$residuals
rmse <- sqrt(mean(model_reg2$residuals^2))
rmse


####  2.799042 RMSE for actual - predictions$fit

# Regression line for data
ggplot(data = cal, aes(Calories.Consumed, log(Weight.gained..grams.)) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))




# LOG() gives the lowest RMSE so we  pick that  our model for train test split

# Data Partition

library(caTools)
set.seed(0)
split <- sample.split(del$Delivery.Time, SplitRatio = 0.8)# --------- both are X_train , X_test
train <- subset(del, split == TRUE)
test <- subset(del, split == FALSE)

######      Model building on Log technique
#IN LOG WAY :#  input = log(x); output = y

model_log <- lm(del$Delivery.Time ~ log(del$Sorting.Time),data=train)

model_log

summary(model_log)

confint(model_log,level=0.95)

model_log$residuals
rmse_log <-  sqrt(mean(model_log$residuals^2))
rmse_log #2.733171

# Prediction on Train data
predict_log <- predict(model_log, interval = "confidence", newdata = train)

pred <- as.data.frame(predict_log)

res1 <- del$Delivery.Time - pred$fit
rme_pred_train <- sqrt(mean(res1^2))
rme_pred_train   # 2.733171 for  Actual vs PRedicted

# Another way to get RMSE
model_log$residuals
rme_model_train <- sqrt(mean(model_log$residuals^2))
rme_model_train

# Prediction on Test data
pred_test <- predict(model_log, interval = "confidence", newdata = test)

pred <- as.data.frame(pred_test)

res1 <- del$Delivery.Time - pred$fit
rme_pred_test <- sqrt(mean(res1^2))
rme_pred_test   # 2.733171 for  Actual vs PRedicted

# Another way to get RMSE
model_log$residuals
rme_model_train <- sqrt(mean(model_log$residuals^2))
rme_model_train

