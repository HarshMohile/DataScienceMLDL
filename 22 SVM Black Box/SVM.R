#####Support Vector Machines 

# Load the Dataset
letter <- read.csv(file.choose())

summary(letter)

# Partition Data into train and test data
letters_train <- letter[1:16000, ]
letters_test  <- letter[16001:20000, ]

# Training a model on the data ----
# Begin by training a simple linear SVM
install.packages("kernlab")
library(kernlab)

letter_classifier <- ksvm(letter ~ ., data = letters_train, kernel = "vanilladot")
?ksvm

## Evaluating model performance ----
# predictions on testing dataset
letter_predictions <- predict(letter_classifier, letters_test)

table(letter_predictions, letters_test$letter)
agreement <- letter_predictions == letters_test$letter
table(agreement)
prop.table(table(agreement))

## Improving model performance ----
letter_classifier_rbf <- ksvm(letter ~ ., data = letters_train, kernel = "rbfdot")
letter_predictions_rbf <- predict(letter_classifier_rbf, letters_test)
agreement_rbf <- letter_predictions_rbf == letters_test$letter
table(agreement_rbf)
prop.table(table(agreement_rbf))
