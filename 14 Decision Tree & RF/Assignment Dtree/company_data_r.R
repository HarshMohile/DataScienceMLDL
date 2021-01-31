library(readr)
library(C50)
library(tree)
library(gmodels)
#install.packages("party")
library(party)
library(caret)

comp <- read.csv(file.choose())

#Decision trees work with continuous variables as well. 
#The way they work is by principle of reduction of variance.

##Exploring and preparing the data ---- Sales is our target variable
str(comp)

table(comp$ShelveLoc)

###   Bad   Good Medium 
###   96     85    219 

hist(comp$Sales)

##  dtree requires Dependant or Categorical to be in Factor levels "A" "b" "Ab"
comp$Urban <-  as.factor(comp$Urban)
comp$US <-  as.factor(comp$US)
comp$ShelveLoc <-  as.factor(comp$ShelveLoc)

str(comp)

## converting sales which is continuous data into categorical 
## then categorical into  Factor levels as input for dtree

sales_distr = ifelse(comp$Sales<10, "No", "Yes")
CD = data.frame(comp, sales_distr)

# again  converting the sales_dist col into factor levels
CD$sales_distr<-  as.factor(CD$sales_distr)

CD <-  CD[,2:12]

library(caTools)
set.seed(0)
split <- sample.split(CD$sales_distr, SplitRatio = 0.8)# --------- both are X_train , X_test
comp_train <- subset(CD, split == TRUE)
comp_test <- subset(CD, split == FALSE)

summary(comp_test)

prop.table(table(CD$sales_distr))
# proportion of NO      Yes are     No    Yes 
#             0.8025 0.1975  respectively


######################## Decision tree Model Building and Eval######################
library(C50) 
#                    Xtrain               y_Train
comp_model <- C5.0(comp_train[, -11], comp_train$sales_distr)
comp_model

# visualization of tree

windows()
plot(comp_model) 

op_tree = ctree(sales_distr ~ CompPrice + Income + Advertising + Population + Price + ShelveLoc
                + Age + Education + Urban + US, data = comp_train)
summary(op_tree)

plot(op_tree)


# Predict(x_test) .first we predict from test data and our comp_model

predictions <- predict(comp_model, comp_test)

# model Accuracy  To find accuracy for the actual  observation  comp_train$sales_distr to prediction
# (x_test,predictions)
test_acc <- mean(comp_train$sales_distr == predictions)
test_acc
# our model is 66 % accurate 0.665625



#################################### Random Forest  in R ##################################
install.packages("randomForest")
library(randomForest)


rfc <- randomForest(sales_distr ~ CompPrice + Income + Advertising + Population + Price + ShelveLoc
                    + Age + Education + Urban + US, data = comp_train)


rfc
#
#Confusion matrix:
#  No Yes class.error
#No  247  10  0.03891051
#Yes  33  30  0.52380952

plot(rfc)
 
