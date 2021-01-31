# classification is about predicting a label and regression is about predicting a quantity

library(readr)
library(e1071)

diabetes <- read.csv(file.choose())

##Exploring and preparing the data ----
str(movies)

names(diabetes)[names(diabetes) == "Class.variable"] <- "Diabetes_check"

####### EDA 
boxplot(diabetes$Age..years.)
skewness(diabetes$Age..years.) #1.12 Right Skewed

diabetes$Diabetes_check <- as.factor(diabetes$Diabetes_check)

str(diabetes)
plot(diabetes)

library(caTools)
set.seed(0)
set.seed(0)
split <- sample.split(diabetes$Diabetes_check, SplitRatio = 0.8)# --------- both are X_train , X_test
db_train <- subset(diabetes, split == TRUE)
db_test <- subset(diabetes, split == FALSE)

summary(db_train)

prop.table(table(diabetes$Diabetes_check))

#
#NO       YES 
#0.6510417 0.3489583 

# visualization on Number of patients  vs non patients by age
library(dplyr)
library(ggplot2)

names(diabetes)

# create a new dataset "age" in  which we  group by age and if they are diabetic with help of diabetec_check varaible

ages<- diabetes %>% group_by(as.factor(Age..years.),as.factor(Diabetes_check)) %>% summarise(Diabetes_check = n())
names(ages)<-c("Age","Class","NumberOfPeople")


# ggplot(data,aes(x,y))+
# geom(aes( hue="Class"))+
#scale_x_discrete
#labels(title)

ggplot(ages,aes(Age,NumberOfPeople))+geom_bar(aes(fill = Class), position = "dodge", stat='identity')+ 
 labs(title="Number of Diabeteics vs Non-Diabetics by Age")

# scale_x_discrete(breaks=seq(21, 81, 2))+
############## ADABOOSTING #####################

library(adabag)

ab_model <- boosting(Diabetes_check ~ ., data = db_train, boos = TRUE)

### Model Eval on test data   model,X_test
adaboost_test <- predict(ab_model, db_test)

table(adaboost_test$class, db_test$Diabetes_check)

#    NO YES
#NO  82  24
#YES 18  30
 # prediction$class  compared with actual observed real value to get acc
mean(adaboost_test$class == db_test$Diabetes_check)
#0.7272727

### Model Eval on Train data   model,X_train

adaboost_train <- predict(ab_model, db_train)

table(adaboost_train$class, db_train$Diabetes_check)

#     NO YES
#NO  400   0
#YES   0 214

mean(adaboost_train$class == db_train$Diabetes_check)
# 100 % accuracy on training data ut 70% on test data