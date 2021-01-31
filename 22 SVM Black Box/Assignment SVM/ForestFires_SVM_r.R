# Forest Fires SVM
library(readr)
#install.packages("dummies")
library(dummies)
library(ggplot2)
library(dplyr)
library(caret)

setwd('D:\\360Assignments\\Submission')
# Load the Dataset
ff <- read.csv(file.choose())

ff <- ff[,c(-1,-2)] # dropped the first 2 cols

ff$size_category <- factor(ff$size_category)

# Splitting the train test 
library(caTools)
set.seed(0)
split <- sample.split(ff$size_category, SplitRatio = 0.8)# --------- both are X_train , X_test
train <- subset(ff, split == TRUE)
test <- subset(ff, split == FALSE)

library(e1071)

svm.model <- svm(size_category ~ .,data=train ,
                 type='C-classification',
                 kernel='linear',
                 scale=FALSE)

summary(svm.model)

# get parameter of the hyperplane 
w <- t(svm.model$coefs) %*% svm.model$SV  # weights
b <- svm.model$rho   # negative intercepts

## Evaluating model performance ----
# predictions on testing dataset
predictions <- predict(svm.model, test)

table(predictions, test$size_category)

'
predictions large small
      large    26     0
      small     2    76
'

accuracy <- predictions == test$size_category


table(accuracy)
'
accuracy
FALSE  TRUE 
    2   102  
'

prop.table(table(accuracy))
'
     FALSE       TRUE 
0.01923077 0.98076923 
'


# with kernel as RB
svm.radial <- svm(size_category ~ .,data=train ,
                  type='C-classification',
                  kernel='radial',
                  scale=FALSE)

summary(svm.radial)

# get parameter of the hyperplane 
w <- t(svm.radial$coefs) %*% svm.radial$SV  # weights
b <- svm.radial$rho   # negative intercepts

## Evaluating model performance ----
# predictions on testing dataset
predictions <- predict(svm.radial, test)

table(predictions, test$size_category)

'
predictions large small
      large     2     0
      small    26    76

'

accuracy <- predictions == test$size_category


table(accuracy)
'
accuracy
FALSE  TRUE 
   26    78 
'

prop.table(table(accuracy))
'
accuracy
FALSE  TRUE 
 0.25  0.75 
'

