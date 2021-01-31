library(readr)
library(gmodels)
library(caTools)
install.packages("caret")
library(caret)
install.packages("pROC")
library(pROC)
install.packages("mlbench")
library(mlbench)
install.packages("lattice")
library(lattice)
install.packages("gmodels")
library(gmodels)
install.packages("class")
library(class)

glass <- read.csv(file.choose())

#************************ EDA **************************
# Remove the last col (type ) which is our TARGET variable to predict
gl <- glass[,c(-10)]

table(glass$Type)
"
 1  2  3  5  6  7 
70 76 17 13  9 29 
"
#structure of glass dataset
str(glass$Type)

#factor the typecolumn
glass$Type <- as.factor(glass$Type)


#structure of glass dataset AFTER factoring give 6 levels 
str(glass$Type)

#Rouding the proportion of each Type in glass dataset
round(prop.table(table(glass$Type)) * 100, digits = 6)

#1         2         3         5         6         7 
#32.710280 35.514019  7.943925  6.074766  4.205607 13.551402 


summary(glass[c("RI", "Na", "Mg")])

## preprocessing the data (normalizing the data)

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}


gl_n <- as.data.frame(lapply(gl, normalize))

######### train test split using caTools 
#glass_n <- cbind(glass$Type,glass_n[1:9])

# X dataset
set.seed(123)
train_ind <- sample(2, nrow(gl_n), replace = TRUE, prob = c(0.7,0.3))
glass_train <- gl_n[train_ind==1,]
glass_test <-  gl_n[train_ind==2,]

# y dataset
set.seed(123)
ind1 <- sample(2, nrow(glass), replace = TRUE, prob = c(0.7,0.3))
glass_train_labels <- glass[ind1==1,10]
glass_test_labels <-  glass[ind1==2,10]


# Build a KNN model on dataset knn( X_train , X_test ,  y_train)

predictions <- knn(train = glass_train, test = glass_test, cl = glass_train_labels, k=3)


# ----------- Model Evaluations
table(predictions,glass_test_labels)

#Accuracy 0.6842105
mean(predictions==glass_test_labels)

CrossTable(x=glass_test_labels,y=predictions,prop.chisq = FALSE)


### Data visualizations on KNN

pred.train <- NULL
pred.val <- NULL
error_rate.train <- NULL
error_rate.val <- NULL
accu_rate.train <- NULL
accu_rate.val <- NULL
accu.diff <- NULL
error.diff <- NULL

# pred.train =knn(X_train , X_train, y_train)
# pred_value = knn(X_train , X_test, y_train)

#error = mean( prediction != y_train) 
#error.val=mean( prediction != y_test)

#acc = mean( prediction == y_train)
#acc.val= mean( prediction == y_test)

#Diff
#acc.diff = acc -acc.val
#error.diff = error -error.val


# knn acc and error
#knn_error_df = as df cbind[ all data from  above about error as cols]
#knn_acc_df =  as df cbind[ all data from  above about acc as cols]


for (i in 1:39) {
  pred.train <- knn(train = glass_train, test = glass_train, cl = glass_train_labels, k = i)
  pred.val <- knn(train = glass_train, test = glass_test, cl = glass_train_labels, k = i)
  
  error_rate.train[i] <- mean(pred.train!=glass_train_labels)
  error_rate.val[i] <- mean(pred.val != glass_test_labels)
  
  accu_rate.train[i] <- mean(pred.train == glass_train_labels)
  accu_rate.val[i] <- mean(pred.val == glass_test_labels)  
  
  accu.diff[i] = accu_rate.train[i] - accu_rate.val[i]
  error.diff[i] = error_rate.val[i] - error_rate.train[i]
}

knn.error <- as.data.frame(cbind(k = 1:39, error.train = error_rate.train, error.val = error_rate.val, error.diff = error.diff))
knn.accu <- as.data.frame(cbind(k = 1:39, accu.train = accu_rate.train, accu.val = accu_rate.val, accu.diff = accu.diff))

library(ggplot2)
errorPlot = ggplot() + 
  geom_line(data = knn.error[, -c(3,4)], aes(x = k, y = error.train), color = "blue") +
  geom_line(data = knn.error[, -c(2,4)], aes(x = k, y = error.val), color = "red") +
  geom_line(data = knn.error[, -c(2,3)], aes(x = k, y = error.diff), color = "black") +
  xlab('knn') +
  ylab('ErrorRate')
errorPlot


# knn lowest error rate occurs when  k = 13

accuPlot = ggplot() + 
  geom_line(data = knn.accu[,-c(3,4)], aes(x = k, y = accu.train), color = "blue") +
  geom_line(data = knn.accu[,-c(2,4)], aes(x = k, y = accu.val), color = "red") +
  geom_line(data = knn.accu[,-c(2,3)], aes(x = k, y = accu.diff), color = "black") +
  xlab('knn') +
  ylab('AccuracyRate')

accuPlot


# Build a KNN model on dataset knn( X_train , X_test ,  y_train) on k=13

predictions <- knn(train = glass_train, test = glass_test, cl = glass_train_labels, k=13)


# ----------- Model Evaluations
table(predictions,glass_test_labels)

#Accuracy 0.6842105
mean(predictions==glass_test_labels)
#0.6491228
CrossTable(x=glass_test_labels,y=predictions,prop.chisq = FALSE)

#Accuracy increased to 70 %
