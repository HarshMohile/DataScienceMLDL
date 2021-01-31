library(readr)

sal <- read.csv(file.choose())

# Exploratory data analysis
summary(sal)
describe(sal)

library(Hmisc)
??Hmisc

is.null(sal)
sum(is.na(sal))

library("lattice") # dotplot is part of lattice package

#  Dotplot                               Graphical exploration
dotplot(sal$YearsExperience, main = "Dot Plot of YearExperience")
dotplot(sal$Salary, main = "Dot Plot of Salary")

# boxplot
boxplot(sal$Salary, col = "dodgerblue4")
boxplot(sal$YearsExperience, col = "red", horizontal = T)

# histogram
hist(del$Salary)
hist(del$YearsExperience)


# Normal QQ plot   (Regression Line) for weight gained
qqnorm(sal$Salary)
qqline(sal$Salary)

# Normal QQ plot for Cal consumed
qqnorm(sal$YearsExperience)
qqline(sal$YearsExperience)


# Bivariate analysis
# Scatter plot
plot(sal$YearsExperience, sal$Salary, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "calories consumed", 
     ylab = "Weight gained", pch = 20)  # plot(x,y)
## slight regression line forming a correlation line 


## 0.98 correaltion formed with these 2 cols 
library(corrplot)
cor <- cor(sal)
corrplot(cor,method="number")

#################################### Linear Regression model #################################
model <- lm(Salary ~ YearsExperience, data = sal) # Y ~ X
model

model$residuals
residuals_rmse <- sqrt(mean(model$residuals^2))  
residuals_rmse

####  RMSE recorded as 5592.044


pred <- predict(model,interval = "predict")
pred <- as.data.frame(pred)


#reg_slr <-  sal$Salary - pred$fit
#rmse_slr <- sqrt(mean(reg_slr^2))
#rmse_slr

# ggplot for adding Regression line for data
library(ggplot2)

ggplot(data = sal, aes(YearsExperience, Salary) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ x)

cor(pred$fit, sal$Salary) 
# 0.9782416 for predicted value  to the actual observed value

#######################################################################################
##################################### Transformation Techniques


####################################### Non-linear models = Polynomial models
#3.POLY()
#

model_reg2 <- lm(log(sal$Salary) ~ sal$YearsExperience 
                 + I(sal$YearsExperience*sal$YearsExperience), data = sal)
summary(model_reg2)

predlog <- predict(model_reg2, interval = "predict")
pred <- exp(predlog)

pred <- as.data.frame(pred)
cor(pred$fit, sal$Salary)

res2 = sal$Salary - pred$fit
rmse <- sqrt(mean(res2^2))
rmse
## 5391.082  RMSE for poly technique 

model_reg2$residuals
rmse <- sqrt(mean(model_reg2$residuals^2))
rmse


####  0.08219581 RMSE for actual - predictions$fit

# Regression line for data
ggplot(data = sal, aes(sal$YearsExperience, log(sal$Salary)) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))

# POLY method will be used for out model on SLR

# Data Partition

library(caTools)
set.seed(0)
split <- sample.split(sal$Salary, SplitRatio = 0.8)# --------- both are X_train , X_test
train <- subset(sal, split == TRUE)
test <- subset(sal, split == FALSE)

######      Model building on Ply technique
#IN LOG WAY :#  input = log(y) ~ x +I(x*x)

model_poly <- lm(log(sal$Salary) ~ sal$YearsExperience+I(sal$YearsExperience*sal$YearsExperience),data=train)

model_poly

summary(model_log)

confint(model_poly,level=0.95)

model_poly$residuals
rmse_poly <-  sqrt(mean(model_poly$residuals^2))
rmse_poly #0.08219581

# Prediction on Train data
predict_poly <- predict(model_poly, interval = "confidence", newdata = train)

predict_poly <- exp(predict_poly)
pred <- as.data.frame(predict_poly)


res1 <- train$Salary - pred$fit
rme_pred_train <- sqrt(mean(res1^2))
rme_pred_train   # 33711.97 for  Actual vs PRedicted



# Prediction on Test data
pred_test <- predict(model_poly, interval = "confidence", newdata = test)

pred <- exp(pred_test)
pred <- as.data.frame(pred)

res1 <- test$Salary - pred$fit
rme_pred_test <- sqrt(mean(res1^2))
rme_pred_test   # 33494.24 for  Actual vs PRedicted

# Test error is slightly higher than  train error .slight case of overfit the train data



