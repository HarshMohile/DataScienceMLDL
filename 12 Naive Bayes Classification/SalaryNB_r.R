# Import the salary dataset
library(readr)
install.packages("dummies")
library(dummies)
library(ggplot2)

salary_train <-read.csv(file.choose())
salary_test <- read.csv(file.choose())


ggplot(data=salary_train,aes(x=salary_train$Salary, y = salary_train$age, fill = salary_train$Salary)) +
  geom_boxplot() +
  ggtitle("Box Plot")

salary_train$Salary <- factor(salary_train$Salary)

View(salary_test)

X <- salary_train[ , c(-14)]
y <- as.data.frame(salary_test$Salary)

class(X)
#Converting categorical into dummy variable in X
X_new <- dummy.data.frame(X, sep = ".")

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}



X_new <- normalize(X_new) 
View(X_new)
################# naive bayes 
library(e1071)
install.packages("caret")
library(caret)
install.packages("psych")
library(psych)

model <- naiveBayes(salary_train$Salary ~ .,data=salary_train)
model


y_pred <- predict(model,salary_test)
mean(y_pred==salary_test$Salary)

#confusionMatrix(y_pred,salary_test$Salary)

confusion_test <- table(x = salary_test$Salary, y = y_pred)
confusion_test
#    y
#x         <=50K  >50K
#<=50K  10550   810
#>50K    1911  1789
