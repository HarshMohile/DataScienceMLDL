
# in LRR glmnet package requires X  into matrix form 
library(readr)
toy <- read.csv(file.choose())

names(toy)

toy <- toy[,c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]

# check or Na or null values
str(toy)
sum(toy) # we are able to add since it has no NA values
sum(is.na(toy))
sum(is.null(toy))

# Normal distribution

qqnorm(toy$Price)
qqline(toy$Price)

# Pair plot
pairs(toy) 

# KM and Age has a corr  and its hetroscedasticity
cor(toy)

# Splitting into X and y 
#X =toy[,c(-1)]
#y=toy$Price
library(glmnet)

X <- model.matrix(Price ~ ., data = toy)[,-1]
y <- toy$Price


# we Regularization we use Lambda *slope^2  to reduce sensitivity for x within x
grid <- 10^seq(10, -2, length = 100)
grid

#-------------------------------- Ridge Regression ------------------------------------------------
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
#  0.8604291

predict(model_lasso, s = optimumlambda, type="coefficients", newx = X)

