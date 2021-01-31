library(readr)
library(e1071)

wbcd <- read.csv(file.choose())

wbcd <-  wbcd[,c(-1)]

## EDA 

##Exploring and preparing the data ----
str(wbcd)

wbcd$diagnosis <- as.factor(wbcd$diagnosis)

############## ADABOOSTING #####################

library(adabag)

library(caTools)
set.seed(0)
set.seed(0)
split <- sample.split(wbcd$diagnosis, SplitRatio = 0.8)# --------- both are X_train , X_test
wbcd_train <- subset(wbcd, split == TRUE)
wbcd_test <- subset(wbcd, split == FALSE)


ab_model <- boosting(diagnosis ~ ., data = wbcd_train, boos = TRUE)
ab_model

#attr(,"vardep.summary")
#B   M 
#286 170 

### Model Eval on test data   model,X_test
adaboost_test <- predict(ab_model, wbcd_test)

table(adaboost_test$class, wbcd_test$diagnosis)

#   B  M
#B 71  1
#M  0 41

# prediction$class  compared with actual observed real value to get acc
mean(adaboost_test$class == wbcd_test$diagnosis)
#0.9911504

### Model Eval on Train data   model,X_train

adaboost_train <- predict(ab_model, wbcd_train)

table(adaboost_train$class, wbcd_train$diagnosis)

#    B   M
#B 286   0
#M   0 170

mean(adaboost_train$class == wbcd_train$diagnosis)
# 100 % accuracy on training data and 0.9911504 on test data
