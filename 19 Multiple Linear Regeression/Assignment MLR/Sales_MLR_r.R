# Muliple Linear regression
library(readr)
sales <- read.csv(file.choose())
View(sales)

attach(sales)


# Removing the first column which is numerical index 'X'
sales <- sales[,c(-1)]

names(sales)
# Dummy variable for discrete  categorical data in dataset since it has only 2 category Y/N
sales$cd <- ifelse(sales$cd == 'yes', 1, 0)
#sales$cd <- as.factor(sales$cd)
class(sales$cd)

sales$multi <- ifelse(sales$multi == 'yes', 1, 0)
#sales$multi <- as.factor(sales$multi)
class(sales$multi)

sales$premium <- ifelse(sales$premium == 'yes', 1, 0)
#sales$premium <- as.factor(sales$premium)
class(sales$premium)

# Normal distribution

qqnorm(sales$price)
qqline(sales$price)


summary(sales)

# Scatter plot
plot(sales$price, sales$speed) # Plot relation ships between each X with Y
plot(sales$price, sales$ram)

# Or make a combined plot
pairs(sales)   # Scatter plot for all pairs of variables
plot(sales)

cor(sales$price, sales$speed)
cor(Cars) # correlation matrix

# The Linear Model of interest
model1 <- lm(price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend, data = sales) # lm(Y ~ X)
summary(model1)

model.speed <- lm(price ~ speed)
summary(model.speed)

model.hd <- lm(price ~ hd)
summary(model.hd)

model.adstrend <- lm(price ~ ads + trend)
summary(model.adstrend)

#### Scatter plot matrix with Correlations inserted in graph
# install.packages("GGally")
library(GGally)
ggpairs(sales)


### Partial Correlation matrix
install.packages("corpcor")
library(corpcor)
??corpcor
cor(sales)

cor2pcor(cor(sales))

#----------------------------------- Diagnostic Plots -----------------------------------------
install.packages(car)
library(car)

plot(model1)# Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance
par(mfrow= c(2,2)) ## all above plots in  2 by 2  in one plot
plot(model1)

abline(model1,col="red",lwd=2)

qqPlot(model1, id.n = 5) # QQ plots of studentized residuals, helps identify outliers
#1441 1701

# Deletion Diagnostics for identifying influential obseravations
?influenceIndexPlot

influenceIndexPlot(model1, id.n = 3 ,vars ="Bonf") # Index Plots of the influence measures
influencePlot(model1, id.n = 3) # A user friendly representation of the above

# Regression after deleting the 1441 and 1701 observation row
sales <-  sales[,c(-1441)]
sales <- sales[,c(-1701)]

model.sales <- lm(price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend, data = sales)
summary(model.sales)


###--------------------------------------- Variance Inflation Factors
vif(model.sales)  # VIF is > 10 => collinearity

# Regression model to check R^2 on Independent variables
VIFSP <- lm(speed ~ hd + ram + screen + cd + multi + premium + ads + trend +price)
VIFHD <- lm(hd ~  ram + screen + cd + multi + premium + ads + trend +price +speed)
VIFScr <- lm(screen ~ ram  + cd + multi + premium + ads + trend +price +speed + hd)
VIFRAM <- lm(ram ~ cd + multi + premium + ads + trend +price +speed + hd +screen)

summary(VIFSP)
summary(VIFHD)
summary(VIFScr)
summary(VIFRAM)

# VIF of SP
1/(1-0.95)

#### Added Variable Plots ###### to view heteroscedacity or homoscedacsticity based on increase or decrease in variance 
avPlots(model.sales, id.n = 2, id.cex = 0.8, col = "red") # hetroscedasticity observed



# Evaluation Model Assumptions
plot(model.sales)
plot(model.sales$fitted.values, model.sales$residuals)

qqnorm(model.sales$residuals)
qqline(model.sales$residuals)

# Subset selection
# 1. Best Subset Selection
# 2. Forward Stepwise Selection
# 3. Backward Stepwise Selection / Backward Elimination

install.packages("leaps")
library(leaps)
# Best subset selection
lm_best <- regsubsets(price ~ ., data = sales, nvmax = 15)
summary(lm_best)
?regsubsets

summary(lm_best)$adjr2    # Adjusted R Squared as column name hence $adjr2 off summary of regsubset's model
which.max(summary(lm_best)$adjr2)
coef(lm_best, 3) # top 3 coefficients (y-intercept) using Best subset selection

 # forward subset selection
lm_forward <- regsubsets(price ~ ., data = sales, nvmax = 15, method = "forward")
summary(lm_forward)

summary(lm_forward)$adjr2    # Adjusted R Squared as column name hence $adjr2 off summary of regsubset's model
which.max(summary(lm_forward)$adjr2)
coef(lm_forward, 3) # top 3 coefficients (y-intercept) using forward subset selection

# Backward subset selection
lm_backward <- regsubsets(price ~ ., data = sales, nvmax = 15, method = "backward")
summary(lm_backward)

summary(lm_backward)$adjr2    # Adjusted R Squared as column name hence $adjr2 off summary of regsubset's model
which.max(summary(lm_backward)$adjr2)
coef(lm_backward, 3) # top 3 coefficients (y-intercept) using backward subset selection

#--------------------------------------------------------------------------------------------==--
#------------------------------- Data Partitioning

library(caTools)
set.seed(0)
split <- sample.split(sales$price, SplitRatio = 0.8)# --------- both are X_train , X_test
train <- subset(sales, split == TRUE)
test <- subset(sales, split == FALSE)


# Model Training
model_final <- lm(price ~ ., data=train)
summary(model_final)

# Predictions
pred <- predict(model_final, newdata = test)
pred <- as.data.frame(pred)
error <- sales$price - pred # actual - predicted

test.rmse <- sqrt(mean(error**2)) # one way of finding residuals by predict() and then actual -pred


train.rmse <- sqrt(mean(model_final$residuals**2)) #  other way to just get residual off model
train.rmse

# Step AIC
install.packages("MASS")
library(MASS)
stepAIC(model_final)
