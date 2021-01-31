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


wbbcd <-  read.csv(file.choose() )

wbbcd <- wbbcd[,c(-1)]

is.null(wbbcd)
# o Null values

sum(is.na(wbbcd))


str(wbbcd)

####    EDA   
# Checking the Malign and Bening in dataset 
prop.table(table(wbbcd$diagnosis))
#B         M 
#0.6274165 0.3725835 

names(wbbcd)

## Count plot so only x ="diagnosis"
ggplot(wbbcd, aes(wbbcd$diagnosis, fill="diagnosis"))+
  geom_bar()+
  theme_bw() +
  labs(title = "Diagnosis CountPLot")

## begning are above 300 and Melign are on 200 .

#Checking correlation between all cols 
Corr <- cor(wbbcd[, setdiff(names(wbbcd),'diagnosis')])
Corr
corr_df <-  data.frame(Corr)
corrplot(Corr)

 ### Boxplot for Outliers

box_plot <- function(colnames, independant_var, data, dependant_var) {
  g_1 <- ggplot(data = data, aes(y = independant_var, fill = dependant_var)) +
    geom_boxplot() +
    theme_bw() +
    labs( title = paste(colnames,"Outlier Detection", sep =" "), y = colnames) +
    theme(plot.title = element_text(hjust = 0.5))
  
  plot(g_1)
}


for(x in 1:(ncol(wbbcd) -1)){
  box_plot(colnames = names(wbbcd)[x], independant_var = wbbcd[,x], data = wbbcd, dependant_var = wbbcd[,'diagnosis'])
}

## Most outliers are present in Concavity (B) , Symmetry_worst (M), teture_worst(B,M), Dimension_se (B,M)



#### Splitting the train , test X 
library(caTools)
set.seed(0)
split <- sample.split(wbbcd$diagnosis, SplitRatio = 0.8)
wbcd_train <- subset(wbbcd, split == TRUE)
wbcd_test <- subset(wbbcd, split == FALSE)


###@@@@@@@@@@@@@@@@@@@@ working on train data = db_train

y_train  <- wbcd_train$diagnosis=="B"

?model.matrix
# create dummy variables on attributes 
#-1 means remove the newly created Intercept col after doing model.matrix

X_train <- model.matrix(wbcd_train$diagnosis ~ ., data = wbcd_train)
# 'n-1' dummy variables are required, hence deleting the additional variables

X_train
y_train
####@@@@@@@@@@@@@@@@@@@@@@ working on test data = db_test

y_test <- wbcd_test$diagnosis=="B"

# create dummy variables on attributes


X_test <- model.matrix(wbcd_test$diagnosis ~ ., data = wbcd_test)

X_test
y_test

#@@@@@@@@@@@@@@@@@ Converting the test , train data into DMmatrix for compatible input for Xgboost

# DMatrix on train  ( X ,y ) for train  ,same for test ( X, y)

Xmatrix_train <- xgb.DMatrix(data = X_train, label = y_train)
# DMatrix on test 
Xmatrix_test <- xgb.DMatrix(data = X_test, label = y_test)



#@@@@@@@@@@@@@@@@@@ Applying the xgboost algorithm 
library(xgboost)
xg_boosting <- xgboost(data = Xmatrix_train, nround = 50,
                       objective = "multi:softmax", eta = 0.3, 
                       num_class = 2, max_depth = 3)


# Prediction for test data
xgbpred_test <- predict(xg_boosting, Xmatrix_test) ## predict( model , test )

table(y_test, xgbpred_test) ###  confusion_matrix( y_test, pred)
mean(y_test == xgbpred_test) ### Accuracy

# Prediction for train data
xgbpred_train <- predict(xg_boosting, Xmatrix_train)  ## predict( model , train )

table(y_train, xgbpred_train)
mean(y_train == xgbpred_train)


### Test  confusion matrix on testdata ,xgb_model with Accuracy 0.9823009
#     xgbpred_test
#xgbpred_test
#y_test   0  1
#FALSE   41  1
#TRUE     1 70

#  Train  confusion matrix on traindata ,xgb_model with Accuracy 100%
#      xgbpred_train
#xgbpred_train
#y_train   0   1
#FALSE   170   0
#TRUE      0 286
install.packages("DiagrammeR")
library(DiagrammeR)

xgb.plot.multi.trees(feature_names = names(Xmatrix_train), 
                     model = xg_boosting)



