library(readr)
library(C50)
library(tree)
library(gmodels)
#install.packages("party")
library(party)
library(caret)
library(e1071) 

fraud <-  read.csv(file.choose())
head(fraud)

fraud$Undergrad <-  as.factor(fraud$Undergrad)
fraud$Marital.Status <-  as.factor(fraud$Marital.Status)
fraud$Urban <-  as.factor(fraud$Urban)

# Turning income_tax numerical into categorical and concat with  fraud dataframe
income = ifelse(fraud$Taxable.Income<=30000, "Risky", "Good")
frd_data = data.frame(fraud, income)

frd_data <- frd_data[,c(-3)]

# Converting the frd_data  into factor levels.
frd_data$income <-  as.factor(frd_data$income)

str(frd_data)

#Training and test data split X train Xtest
library(caTools)
set.seed(0)
split <- sample.split(frd_data$income, SplitRatio = 0.8)# --------- both are X_train , X_test
fd_train <- subset(frd_data, split == TRUE)
fd_test <- subset(frd_data, split == FALSE)

summary(fd_test)

prop.table(table(frd_data$income))

#      Good     Risky 
#    0.7933333 0.2066667
# only 20 % percent are observed to have risky Taxable_income to be cross checked

######################## Decision tree Model Building and Eval######################
library(C50) 
#                    Xtrain               y_Train
fd_model <- C5.0(fd_train[, -6], fd_train$income)
fd_model

#Classification Tree
#Number of samples: 480  
#Number of predictors: 5  

plot(fd_model)



op_tree = ctree(income ~ ., data= fd_train)
summary(op_tree)
plot(op_tree)

# Predict(x_test) .first we predict from test data and our comp_model
?predict

predictions <- predict(fd_model, fd_test)
predictions
# model Accuracy  To find accuracy for the actual  observation  fd_test$income to prediction
# (x_test,predictions)
test_acc <- mean(fd_test$income == predictions)
test_acc
# test accuracy ::: 0.7916667 79%


#################################### Random Forest  in R ##################################
install.packages("randomForest")
library(randomForest)

#                        y_train         X_train
rfc <- randomForest(income ~ ., data = fd_train)

rfc
#Confusion matrix:
#       Good Risky class.error
#Good   377     4  0.01049869
#Risky   99     0  1.00000000
plot(rfc)


predictions_rf <- predict(rfc, fd_test)
predictions_rf
# model Accuracy  To find accuracy for the actual  observation  comp_train$sales_distr to prediction
# (x_test,predictions)
test_acc_rf <- mean(fd_test$income == predictions_rf)
test_acc_rf
# test accuracy ::: 0.7666667 .Accuracy reduced in Random Forest 

