library(readr)
library(gmodels)
library(caTools)
library(caret)
library(pROC)
library(mlbench)
library(lattice)
library(gmodels)
library(class)


zoo <- read.csv(file.choose())
zoo1 <- zoo[,c(-1)]

table(zoo1$type)
#  type is our target class and there are 7 categories or types in it


# Freq distr of each type 
round(prop.table(table(zoo$type))*100, digits = 1)
 
#1    2    3    4    5    6    7 
#40.6 19.8  5.0 12.9  4.0  7.9  9.9 

# summary of animal who is type chicken or bird type
summary(zoo[c("eggs","milk","airborne")])

normalize_data <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}

zoo_n <- as.data.frame(lapply(zoo1[1:16], normalize_data))


summary(zoo_n[c("feathers","aquatic","legs")])

# Splitting the data in X,y train
X_train <- zoo_n[1:80,]
X_test <- zoo_n[81:101,]

y_train_labels <- zoo1[1:80,17]

y_test_labels <- zoo1[81:101,17]

# Model Building

library(class)
 # knn(X_train , X_test,  y_train)

pred <- knn(train = X_train, test = X_test, cl = y_train_labels, k=5)
pred


library(gmodels)
# predict(y_test, prediction)

CrossTable(y_test_labels, pred)


#Accuracy 0.7619048
mean(pred==y_test_labels)


## choosing correct value of k 
pred.train <- NULL
pred.val <- NULL
error_rate.train <- NULL
error_rate.val <- NULL
accu_rate.train <- NULL
accu_rate.val <- NULL
accu.diff <- NULL
error.diff <- NULL


for (i in 1:39) {
  pred.train <- knn(train = X_train, test = X_train, cl = y_train_labels, k = i)
  pred.val <- knn(train = X_train, test = X_test, cl = y_train_labels, k = i)
  
  error_rate.train[i] <- mean(pred.train!=y_train_labels)
  error_rate.val[i] <- mean(pred.val != y_test_labels)
  
  accu_rate.train[i] <- mean(pred.train == y_train_labels)
  accu_rate.val[i] <- mean(pred.val == y_test_labels)  
  
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


# knn lowest error rate occurs when  k = 10

accuPlot = ggplot() + 
  geom_line(data = knn.accu[,-c(3,4)], aes(x = k, y = accu.train), color = "blue") +
  geom_line(data = knn.accu[,-c(2,4)], aes(x = k, y = accu.val), color = "red") +
  geom_line(data = knn.accu[,-c(2,3)], aes(x = k, y = accu.diff), color = "black") +
  xlab('knn') +
  ylab('AccuracyRate')

accuPlot


# Build a KNN model on dataset knn( X_train , X_test ,  y_train) on k=13

predictions <- knn(train = X_train, test = X_test, cl = y_train_labels, k=10)


# ----------- Model Evaluations
table(predictions,y_test_labels)

#Accuracy 0.6842105
mean(predictions==y_test_labels)
#[1] 0.8095238
CrossTable(x=y_test_labels,y=predictions,prop.chisq = FALSE)




