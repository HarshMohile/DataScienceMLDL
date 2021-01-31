# SVM on Salary Dataset
library(readr)
#install.packages("dummies")
library(dummies)
library(ggplot2)
library(dplyr)
library(caret)

setwd('D:\\360Assignments\\Submission')
# Load the Dataset
sal_train <- read.csv(file.choose())
sal_test <-  read.csv(file.choose())


# Data Viz
# for Training data Sal response with Age as predictor
ggplot(data=sal_train,aes(x=sal_train$Salary, y = sal_train$age, fill = sal_train$Salary)) +
  geom_boxplot() +
  ggtitle("Box Plot")

# for test data Sal response with Age as predictor
ggplot(data=sal_test,aes(x=sal_test$Salary, y = sal_test$age, fill = sal_test$Salary)) +
  geom_boxplot() +
  ggtitle("Box Plot")

#Check for NA
anyNA(sal_train)
anyNA(sal_test)



#Converting categorical into dummy variable in sal_train and sal_test

sal_train$Salary <- factor(sal_train$Salary)
sal_test$Salary <- factor(sal_test$Salary)


library(e1071)

svm.model <- svm(Salary ~ .,data=sal_train ,
                            type='C-classification',
                             kernel='linear',
                              scale=FALSE)

summary(svm.model)


# get parameter of the hyperplane 
w <- t(svm.model$coefs) %*% svm.model$SV  # weights
b <- svm.model$rho   # negative intercepts

## Evaluating model performance ----
# predictions on testing dataset
predictions <- predict(svm.model, sal_test)

table(predictions, sal_test$Salary)

'
predictions  <=50K  >50K
      <=50K   9410  1557
      >50K    1950  2143
'

accuracy <- predictions == sal_test$Salary


table(accuracy)
'
FALSE  TRUE 
 3507 11553 
'

prop.table(table(accuracy))
'
    FALSE      TRUE 
0.2328685 0.7671315 
'


# with kernel as RB
svm.radial <- svm(Salary ~ .,data=sal_train ,
                 type='C-classification',
                 kernel='radial',
                 scale=FALSE)

summary(svm.radial)

# get parameter of the hyperplane 
w <- t(svm.radial$coefs) %*% svm.radial$SV  # weights
b <- svm.radial$rho   # negative intercepts

## Evaluating model performance ----
# predictions on testing dataset
predictions <- predict(svm.radial, sal_test)

table(predictions, sal_test$Salary)

'
predictions  <=50K  >50K
      <=50K  10854  1568
      >50K     506  2132

'

accuracy <- predictions == sal_test$Salary


table(accuracy)
'
FALSE  TRUE 
 2074 12986
'

prop.table(table(accuracy))
'
    FALSE      TRUE 
0.1377158 0.8622842
'

