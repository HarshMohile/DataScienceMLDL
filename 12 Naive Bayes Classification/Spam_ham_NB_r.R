library(readr)
library(dplyr)
library(tm)
install.packages("caTools")
messages <- read.csv(file.choose())
View(messages)


?factor# to factor the 2 types of messages (ham,spam) otherwise prediction will return with list 0 objects
messages$type <- factor(messages$type)

messages %>%
  filter(type == "ham") %>%
  select(nrow(messages$text))
# data frame with 0 columns and 4812 rows in ham
messages %>%
  filter(type == "spam") %>%
  select(nrow(messages$text))
#data frame with 0 columns and 747 rows in spam

prop.table(table(messages$type))
# 86% ham and 13% spam in messages data

# converting a character to > corpus 
library(tm)

str(messages$text)
messages_corpus <- Corpus(VectorSource(messages$text))
View(messages_corpus)

messages_corpus <- tm_map(messages_corpus, function(x) iconv(enc2utf8(x), sub='byte'))

str(messages_corpus)
class(messages_corpus)

### Data Cleaning ###
corpus_clean <- tm_map(messages_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, stripWhitespace)


# create a document-term sparse matrix
sms_dtm <- DocumentTermMatrix(corpus_clean)
sms_dtm
View(sms_dtm[1:10, 1:30])

# To view DTM we need to convert it into matrix first
dtm_matrix <- as.matrix(sms_dtm)
str(dtm_matrix)

View(dtm_matrix[1:10, 1:20])

colnames(sms_dtm)[1:50]

# creating training and test datasets
sms_raw_train <- messages[1:4169, ]
sms_raw_test  <- messages[4170:5559, ]

sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test  <- corpus_clean[4170:5559]

sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test  <- sms_dtm[4170:5559, ]

# check that the proportion of spam is similar
prop.table(table(messages$type))

prop.table(table(sms_raw_train$type))
prop.table(table(sms_raw_test$type))

# indicator features for frequent words
# dictionary of words which are used more than 5 times
sms_dict <- findFreqTerms(sms_dtm_train, 5)

sms_train <- DocumentTermMatrix(sms_corpus_train, list(dictionary = sms_dict))
sms_test  <- DocumentTermMatrix(sms_corpus_test, list(dictionary = sms_dict))

sms_test_matrix <- as.matrix(sms_test)
View(sms_test_matrix[1:10,1:10])

# convert counts to a factor
# custom function: if a word is used more than 0 times then mention 1 else mention 0
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

# apply() convert_counts() to columns of train/test data
# Margin = 2 is for columns
# Margin = 1 is for rows
sms_train <- apply(sms_train, MARGIN = 2, convert_counts)
sms_test  <- apply(sms_test, MARGIN = 2, convert_counts)
?apply()

View(sms_test[1:10,1:10])

##  Training a model on the data ----
install.packages("e1071")
library(e1071)

sms_classifier <- naiveBayes(sms_train, sms_raw_train$type)
sms_classifier

##  Evaluating model performance

sms_test_pred <- predict(sms_classifier, sms_test)

library(gmodels)
CrossTable(sms_test_pred, sms_raw_test$type,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))


# ON Test Data
test_acc = mean(sms_test_pred == sms_raw_test$type)
test_acc

# On Training Data
sms_train_pred <- predict(sms_classifier, sms_train)

train_acc = mean(sms_train_pred == sms_raw_train$type)
train_acc

hist(train_acc)

