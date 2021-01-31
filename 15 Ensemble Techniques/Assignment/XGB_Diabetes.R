library(readr)
library(caret) #ML Model buidling package
library(tidyverse) #ggplot and dplyr
library(MASS) #Modern Applied Statistics with S
library(mlbench) #data sets from the UCI repository.
library(summarytools)
library(corrplot) #Correlation plot
library(gridExtra) #Multiple plot in single grip space
library(timeDate) 
library(pROC) #ROC
library(caTools) #AUC
library(rpart.plot) #CART Decision Tree
library(e1071) #imports graphics, grDevices, class, stats, methods, utils
library(graphics) #fourfoldplot

db <- read.csv(file.choose())

str(db)

names(db)[names(db)=="Class.variable"] <- "Diabetic_check"

#### Splitting the train , test X 
library(caTools)
set.seed(0)
split <- sample.split(db$Diabetic_check, SplitRatio = 0.8)
db_train <- subset(db, split == TRUE)
db_test <- subset(db, split == FALSE)


summary(db_train)

prop.table(table(db$Diabetic_check))

#NO       YES 
#0.6510417 0.3489583 


## Descriptive Statistics 
summarytools::descr(db)


### EDA
## Count plot so only x ="diabete_Check"
ggplot(db, aes(db$Diabetic_check, fill="Diabetic_check"))+
  geom_bar()+
  theme_bw() +
  labs(title = "Diabetes CountPLot")
## Non diabeteic are in more in number

## Checking correlation btw  x=all rows  and y =Diabetic_check  
#.setdiff(names(db),'Diabetic_check') operates row wise on  cols mentioned names(db) ,Diabetic_check

Corr <- cor(db[, setdiff(names(db),'Diabetic_check')])
Corr

# Correlation matrix plots
corrplot(Corr)



#### working with outliers by iterating the cols like n TWSS for number of cols for(x in cols)

box_plot <- function(colnames, independant_var, data, dependant_var) {
    g_1 <- ggplot(data = data, aes(y = independant_var, fill = dependant_var)) +
      geom_boxplot() +
      theme_bw() +
      labs( title = paste(colnames,"Outlier Detection", sep =" "), y = colnames) +
      theme(plot.title = element_text(hjust = 0.5))
    
    plot(g_1)
}

for (x in 1:(ncol(db_train)-1)) {
  box_plot(colnames = names(db_train)[x], independant_var = db_train[,x], data = db_train, dependant_var = db_train[,'Diabetic_check'])
}

## most outliers occured in [ X2 hour insulin, pedigree ,plasma glucose ]
#and  its corr with [diabetes_check] column
## ouliers handled by CARET package

# install.packages("xgboost")
library(xgboost)


 ###@@@@@@@@@@@@@@@@@@@@ working on train data = db_train

y_train  <- db_train$Diabetic_check == "YES"

?model.matrix
# create dummy variables on attributes 
#-1 means remove the newly created Intercept col after doing model.matrix

X_train <- model.matrix(db_train$Diabetic_check ~ .-1, data = db_train)
# 'n-1' dummy variables are required, hence deleting the additional variables

X_train
y_train
####@@@@@@@@@@@@@@@@@@@@@@ working on test data = db_test

y_test <- db_test$Diabetic_check == "YES"

# create dummy variables on attributes


X_test <- model.matrix(db_test$Diabetic_check ~ .-1, data = db_test)

X_test
y_test

#@@@@@@@@@@@@@@@@@ Converting the test , train data into DMmatrix for compatible input for Xgboost

# DMatrix on train  ( X ,y ) for train  ,same for test ( X, y)

Xmatrix_train <- xgb.DMatrix(data = X_train, label = y_train)
# DMatrix on test 
Xmatrix_test <- xgb.DMatrix(data = X_test, label = y_test)


#@@@@@@@@@@@@@@@@@@ Applying the xgboost algorithm 

xg_boosting <- xgboost(data = Xmatrix_train, nround = 50,
                       objective = "multi:softmax", eta = 0.3, 
                       num_class = 2, max_depth = 100)


# Prediction for test data
xgbpred_test <- predict(xg_boosting, Xmatrix_test) ## predict( model , test )

table(y_test, xgbpred_test) ###  confusion_matrix( y_test, pred)
mean(y_test == xgbpred_test) ### Accuracy

# Prediction for train data
xgbpred_train <- predict(xg_boosting, Xmatrix_train)  ## predict( model , train )

table(y_train, xgbpred_train)
mean(y_train == xgbpred_train)


### Test  confusion matrix on testdata ,xgb_model with Accuracy 0.7402597
#     xgbpred_test
#y_test   0  1
#FALSE    86 14
#TRUE     26 28

#  Train  confusion matrix on traindata ,xgb_model with Accuracy 100%
#      xgbpred_train
#y_train    0   1
#FALSE    400   0
#TRUE       0 214

library(DiagrammeR)

xgb.plot.multi.trees(feature_names = names(Xmatrix_train), 
                     model = xg_boosting)
