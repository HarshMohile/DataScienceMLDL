# in LRR glmnet package requires X  into matrix form 
library(readr)
strt <- read.csv(file.choose())

# Categorical character conversion

#strt$State <- as.factor(strt$State)
#strt <- fastDummies::dummy_cols(strt)
#knitr::kable(strt)


library(dplyr)
install.packages("mlr")
library(mlr)

st1 <- createDummyFeatures(strt$State, cols = "State")
?cbind
strt <- cbind(strt ,st1)
# Now Remove the state column
strt <- strt[,c(-4)]


#df <- read.csv("your_data_path")
#df <- df %>%
#mutate_if(is.factor, as.numeric)


# check or Na or null values
str(strt)
sum(is.na(toy))
sum(is.null(toy))


# Normal distribution

qqnorm(strt$Profit)
qqline(strt$Profit)


plot(x = strt$R.D.Spend ,y =strt$Profit) #  correlation between  y and x -feature (RnD spent)

# Pair plot
pairs(strt) 

# KM and Age has a corr  and its hetroscedasticity
cor(strt)


X <- model.matrix(Profit ~ ., data = strt)[,-4]
y <- strt$Profit



#-------------------------------- Ridge Regression ------------------------------------------------
# we Regularization we use Lambda *slope^2  to reduce sensitivity for x within x
grid <- 10^seq(10, -2, length = 100)
grid

# Since its Ridge Regression we are applying alpha as 0

library(glmnet)
model_ridge <- glmnet(X, y, alpha = 0, lambda = grid)
summary(model_ridge)

# Using CV 
cv_fit <- cv.glmnet(X, y, alpha = 0, lambda = grid)
plot(cv_fit)

# Optimum lambda
optimumlambda <- cv_fit$lambda.min

pred <- predict(model_ridge, s = optimumlambda, newx = X)

# Model Evaluation
sse <- sum((pred-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared 
#   0.8531309 

predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)


#------------------------------------ Lasso Regression --------------------------------------------
# ussing alpha as 1
model_lasso <- glmnet(X, y, alpha = 1, lambda = grid)
summary(model_lasso)

# Using CV
cv_fit_las <- cv.glmnet(X, y, alpha = 1, lambda = grid)
plot(cv_fit_las)

#optimum lambda
optimumlambda_1 <- cv_fit_las$lambda.min

pred_lasso <- predict(model_lasso, s = optimumlambda_1, newx = X)

# Model Evaluation
sse <- sum((pred_lasso-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared
#  0.944867
# high R^2 squared explains the variance  in Y given by x and is good for the model

predict(model_lasso, s = optimumlambda, type="coefficients", newx = X)

