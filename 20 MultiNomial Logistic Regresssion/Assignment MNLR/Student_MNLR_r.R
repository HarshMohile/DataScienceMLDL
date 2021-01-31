# Multinomial Logit Model

require('mlogit')
require('nnet')
library(dplyr)

library(readr)

st <- read.csv(file.choose())

st <- st[,c(-1,-2)]  # removed  unnamed and id col

# Data cleaning 
sum(is.na(st))

# rename the female col to gender
names(st)[names(st) == "female"] <- "gender"

# Factor the categorical variables
st$gender <- as.factor(st$gender)
st$ses <- as.factor(st$ses)
st$schtyp <- as.factor(st$schtyp)
st$honors <- as.factor(st$honors)


####----- -----------------------Splitting the dataset -------------------

library(caTools)
set.seed(0)
split <- sample.split(st$prog, SplitRatio = 0.8)# --------- both are X_train , X_test
train <- subset(st, split == TRUE)
test <- subset(st, split == FALSE)

st_model <- multinom(prog ~ ., data = train)
summary(st_model)

# academic to be baseline in this case
#The levels of a factor are re-ordered so that the level 
#specified by ref is first and the others are moved down

st$prog  <- relevel(st$prog, ref= "academic") 


##### Significance of Regression Coefficients    (z , p , exp(coef), prob   )###

z <- summary(st_model)$coefficients / summary(st_model)$standard.errors

z

p_value <- (1 - pnorm(abs(z), 0, 1)) * 2

p_value


# odds ratio 
exp(coef(st_model))

# check for fitted values on training data (Creates seperate col for each category present in y)
# academic , general , vocation
?fitted

prob <- fitted(st_model)

prob
###############################################TEST DATA##############################################
# Predicted on test data
pred_test <- predict(st_model, newdata =  test, type = "probs") # type="probs" is to calculate probabilities
pred_test

# Find the accuracy of the model
class(pred_test)
pred_test <- data.frame(pred_test)
View(pred_test)
pred_test["prediction"] <- NULL

# Custom function that returns the predicted value based on probability
get_names <- function(i){
  return (names(which.max(i)))
}

 # it just stores the names having max(probab) for that row in a predtest_name
predtest_name <- apply(pred_test, 1, get_names)
?apply
pred_test$prediction <- predtest_name
View(pred_test)


# Reason behind NOT putting prediction straight away in confusionmatrix(prediction, test$prog)
#is because 
#test%prog has actual category string and prediction <- predict() gives probab 
#.so found the max probab through the custom function  (predtest_name)
#that gave o/p as category string "academic" etc and used that to compare to test$prog


# Confusion matrix
table(predtest_name, test$prog)

# confusion matrix Visualization
# barplot(table(predtest_name, test$choice), beside = T, col =c("red", "lightgreen", "blue", "orange"), legend = c("bus", "car", "carpool", "rail"), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")

barplot(table(predtest_name, test$prog),
                  beside = T, col =c("red", "lightgreen", "blue", "orange"),
                  main = "Predicted(X-axis) - Legends(Actual)", 
                  ylab ="count")

# Accuracy on test data 0.55
mean(predtest_name == test$prog)

############################################TRAINING DATA#########################################
#training data
pred_train <- predict(st_model, newdata =  train, type = "probs") # type="probs" is to calculate probabilities
pred_train

# Accuracy of the model 
class(pred_train)
pred_traindf <- as.data.frame(pred_train) # convert into df

pred_traindf["PREDICTIONS"] <- NULL

# Custom func to store the max prob name for that spec col
# it just stores the names having max(probab) for that row in a predtest_name
get_names <- function(i){
  return (names(which.max(i)))
}

 # Apply( var, margin, function)

predtrain_name <- apply(pred_traindf, 1, get_names)
?apply
pred_traindf["PREDICTIONS"] <- predtest_name
View(pred_traindf)

# compare pred with actual
table(pred_traindf$PREDICTIONS , train$prog)

#Plotting the confusion matrix
barplot(table(pred_traindf$PREDICTIONS , train$prog)
        ,beside=T
        ,col =c("red", "lightgreen", "blue", "orange")
        ,main = "Predicted(X-axis) - Legends(Actual) Training data"
        , ylab ="count")

# beside =T (thins the buildings)

acc_train <- pred_traindf$PREDICTIONS == train$prog
mean(acc_train)      
