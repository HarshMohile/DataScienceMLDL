library(readr)

emp <- read.csv(file.choose())

# Exploratory data analysis
summary(emp)
describe(emp)

library(Hmisc)
??Hmisc

is.null(emp)
sum(is.na(emp))

library("lattice") # dotplot is part of lattice package

#  Dotplot                               Graphical exploration
dotplot(emp$Salary_hike, main = "Dot Plot of Sal hike")
dotplot(emp$Churn_out_rate, main = "Dot Plot of Churn out rate")

# boxplot
boxplot(emp$Salary_hike, col = "dodgerblue4")
boxplot(emp$Churn_out_rate, col = "red", horizontal = T)

# histogram
hist(emp$Salary_hike)
hist(emp$Churn_out_rate)


# Normal QQ plot   (Regression Line) for weight gained
qqnorm(emp$Salary_hike)
qqline(emp$Salary_hike)

# Normal QQ plot for Cal consumed
qqnorm(emp$Churn_out_rate)
qqline(emp$Churn_out_rate)


# Bivariate analysis
# Scatter plot
plot(emp$Salary_hike, emp$Churn_out_rate, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "Salary_hike", 
     ylab = "Churn_out_rate", pch = 20)  # plot(x,y)
## slight regression line forming a correlation line in a downward slope


## -0.91 correaltion formed with these 2 cols 
library(corrplot)
cor <- cor(emp)
corrplot(cor,method="number")

#################################### Linear Regression model #################################
model <- lm(emp$Salary_hike ~ emp$Churn_out_rate, data = emp) # Y ~ X
model

model$residuals
residuals_rmse <- sqrt(mean(model$residuals^2))  
residuals_rmse
#35.89264

####  RMSE recorded as 35.89264


pred <- predict(model,interval = "predict")
pred <- as.data.frame(pred)


#reg_slr <-  sal$Salary - pred$fit
#rmse_slr <- sqrt(mean(reg_slr^2))
#rmse_slr

# ggplot for adding Regression line for data
library(ggplot2)

ggplot(data = emp, aes(emp$Churn_out_rate, emp$Salary_hike) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ x)

cor(pred$fit, emp$Salary_hike) 
# 0.9117216 for predicted value  to the actual observed value


#######################################################################################
##################################### Transformation Techniques

# 1. LOG ()
# input = log(x); output = y

plot(log(emp$Churn_out_rate), emp$Salary_hike)
cor(log(emp$Churn_out_rate), emp$Salary_hike)


########################### Model building for LOG( )  transformation technique
reg_log <- lm(emp$Salary_hike ~ log(emp$Churn_out_rate), data = emp)
summary(reg_log)


?confint  # Confidence Intervals for Model Parameters
confint(reg_log,level = 0.95)

# pred
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)

cor(pred$fit, emp$Churn_out_rate)  # corr btw predicted'Weigth and actual weigth

rmse <- sqrt(mean(reg_log$residuals^2))
rmse
### 31.06952 ---> RMSE for LM in LOG() technique


# Regression line for data
ggplot(data = emp, aes(log(emp$Churn_out_rate), emp$Salary_hike) ) +
  geom_point(color = 'red') + stat_smooth(method = lm, formula = y ~ log(x))

# 35 rmse for normal SLR and 31 for Log Technique
# We will use Log for train and test data

#=========================Data Partitioning ffor final model =============================
library(caTools)
set.seed(0)
split <- sample.split(emp$Salary_hike, SplitRatio = 0.8)# --------- both are X_train , X_test
train <- subset(emp, split == TRUE)
test <- subset(emp, split == FALSE)

model_log <- lm(log(emp$Salary_hike) ~ emp$Churn_out_rate+I(emp$Churn_out_rate*emp$Churn_out_rate),data=train)

model_log

summary(model_log)

confint(model_log,level=0.95)

model_log$residuals
rmse_log <-  sqrt(mean(model_log$residuals^2))
rmse_log #0.007448004

# Prediction on Train data
predict_train <- predict(model_log, interval = "confidence", newdata = train)


pred <- as.data.frame(predict_train)


res1 <- train$Salary_hike - pred$fit
rme_pred_train <- sqrt(mean(res1^2))
rme_pred_train   # 1662.848 for  Actual vs PRedicted



# Prediction on Test data
pred_test <- predict(model_log, interval = "confidence", newdata = test)


pred <- as.data.frame(pred)

res1 <- test$Salary - pred$fit
rme_pred_test <- sqrt(mean(res1^2))
rme_pred_test   # 1687.932 for  Actual vs PRedicted

# Test error is slightly higher than  train error .slight case of overfit the train data







