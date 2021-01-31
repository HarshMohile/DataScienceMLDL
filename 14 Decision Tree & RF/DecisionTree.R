# Load the Data
# credit.csv

##Exploring and preparing the data ----
str(credit)

# look at the class variable
table(credit$default)

credit$default <- as.factor(credit$default)
credit$checking_balance <- as.factor(credit$checking_balance)

# Shuffle the data
credit_rand <- credit[order(runif(1000)), ]
str(credit_rand)

# split the data frames
credit_train <- credit_rand[1:900, ]
credit_test  <- credit_rand[901:1000, ]

# check the proportion of class variable
prop.table(table(credit_rand$default))
prop.table(table(credit_train$default))
prop.table(table(credit_test$default))

# Step 3: Training a model on the data
install.packages("C50")
library(C50)

credit_model <- C5.0(credit_train[, -17], credit_train$default)

windows()
plot(credit_model) 

# Display detailed information about the tree
summary(credit_model)

# Step 4: Evaluating model performance
# Test data accuracy
test_res <- predict(credit_model, credit_test)
test_acc <- mean(credit_test$default == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(credit_test$default, test_res, dnn = c('actual default', 'predicted default'))

# On Training Dataset
train_res <- predict(credit_model, credit_train)
train_acc <- mean(credit_train$default == train_res)
train_acc

table(credit_train$default, train_res)

