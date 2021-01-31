library(readr)
library(C50)
library(tree)
library(gmodels)
#install.packages("party")
library(party)
library(caret)
library(e1071)  


db <- read.csv(file.choose())

####### EDA 
boxplot(db$Age..years.)
skewness(db$Age..years.) #1.12 Right Skewed

plot(db$Number.of.times.pregnant,db$Number.of.times.pregnant)

# Ggplot for pregnant times v age (heavy corr )
ggplot(data = db, mapping = aes(x = db$Number.of.times.pregnant, y = db$Number.of.times.pregnant))+
  geom_point()
# ggplot for no of times pregnant v having dibates (Class variable) --- (No correlation )
ggplot(data = db, mapping = aes(x = db$Number.of.times.pregnant, y = db$Class.variable))+
  geom_point()

# Rename column where  variable is "Class variable"
names(db)[names(db) == "Class.variable"] <- "Diabetes_check"

db$Diabetes_check <- as.factor(db$Diabetes_check)

str(db)

#Training and test data split
library(caTools)
set.seed(0)
split <- sample.split(db$Diabetes_check, SplitRatio = 0.8)# --------- both are X_train , X_test
db_train <- subset(db, split == TRUE)
db_test <- subset(db, split == FALSE)


summary(db_test)

prop.table(table(db$Diabetes_check))
#NO       YES 
#0.6510417 0.3489583 

######################## Decision tree Model Building and Eval######################
library(C50) 
#                    Xtrain               y_Train
db_model <- C5.0(db_train[, -9], db_train$Diabetes_check)
db_model

#Classification Tree
#Number of samples: 614 
#Number of predictors: 8 

plot(db_model)

names(db)

op_tree = ctree(Diabetes_check ~ ., data= db_train)
summary(op_tree)
plot(op_tree)

# Predict(x_test) .first we predict from test data and our comp_model

predictions <- predict(db_model, db_test)
predictions
# model Accuracy  To find accuracy for the actual  observation  comp_train$sales_distr to prediction
# (x_test,predictions)
test_acc <- mean(db_test$Diabetes_check == predictions)
test_acc
# test accuracy ::: 0.7402597 74%


#################################### Random Forest  in R ##################################
install.packages("randomForest")
library(randomForest)

 #                        y_train         X_train
rfc <- randomForest(Diabetes_check ~ ., data = db_train)

#Confusion matrix:
#     NO YES class.error
#NO  347  53   0.1325000
#YES  81 133   0.3785047
plot(rfc)


predictions_rf <- predict(rfc, db_test)
predictions_rf
# model Accuracy  To find accuracy for the actual  observation  comp_train$sales_distr to prediction
# (x_test,predictions)
test_acc_rf <- mean(db_test$Diabetes_check == predictions_rf)
test_acc_rf
# test accuracy ::: 0.7012987 .Accuracy reduced in Random Forest 

