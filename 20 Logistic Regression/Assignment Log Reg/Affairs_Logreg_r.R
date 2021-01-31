# Logistic regeression On affairs data
library(readr)

setwd('D:\\360Assignments\\Submission')
aff <- read.csv(file.choose())

# Data cleaning 
sum(is.na(aff))


# remove the index col
aff <- aff[,c(-1)]

# See the proportion 
prop.table(table(aff$naffairs))
'
0          1          2          3          7         12 
0.75041597 0.05657238 0.02828619 0.03161398 0.06988353 0.06322795 
Converting above 0 as 1
'
aff$naffairs <- ifelse(aff$naffairs > 0, 1, 0)

prop.table(table(aff$naffairs))

# Model  building on whole data

model <- glm(naffairs ~ ., data = aff, family = "binomial")  # 0,1
summary(model)

# for taking  Log Ratio first take exp() of  coef(model)
exp(coef(model))

# there are NA in coef(Model)


# Check multicollinearity in the model using VIF
library(car) 
vif(model)

#  There are aliased coefficients (NA) in the model so we check the culprits
#the linearly dependent variables

#find the aliased cols
ld.vars <- attributes(alias(model)$Complete)$dimnames[[1]]

#"vryhap"   "vryrel"   "yrsmarr6"
#Remove these colS, [SUBSET FOR DATA]
aff1 <-  subset(aff,select =-c(vryhap,vryrel,yrsmarr6))

# Again model build after removing VIF columns
model <- glm(naffairs ~ ., data = aff1, family = "binomial")  # 0,1
summary(model)

# for taking  Log Ratio first take exp() of  coef(model)
exp(coef(model))

#once again checking for MultiCollinearity  . this none found
library(car) 
vif(model)

# Prediction ( model, data , type)
pred_prob <- predict(model, aff1, type = "response")
pred_prob

# Convert the probabilities to binary output form using cutoff
pred_values <- ifelse(pred_prob > 0.5, 1, 0)

library(caret)
# Confusion Matrix in a detailed format
# confusionMatrix(actual , predicted) but in factor( levels=c(0,1))

confusionMatrix(aff1$naffairs , pred_values)

'
    0   1
0 429 118
1  22  32
'

# Decide on optimal prediction probability cutoff for the model

library(InformationValue)  # (y ,prob_pred)
optCutOff <- optimalCutoff(aff1$naffairs, pred_prob)
optCutOff
#  0.5276987

# Misclassification Error - the percentage mismatch of predicted vs actuals
# Lower the misclassification error, better the model.

misClassError(aff1$naffairs, pred_prob, threshold = optCutOff)

#0.223

# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(aff1$naffairs, pred_prob)

# ROC curve with AUROC  gave about 0.72

# Confusion Matrix ( wrt to cutoff ). ::  to classify 0 or 1 based on Optimalcutoff)
pred_cutoff <- ifelse(pred_prob > optCutOff, 1, 0)

results <- confusionMatrix(pred_cutoff, aff1$naffairs)

'
    0  1
0 440 11
1 123 27
'
sensitivity(pred_cutoff, aff1$naffairs)   
#0.7105263

####----- -----------------------Splitting the dataset -------------------

library(caTools)
set.seed(0)
split <- sample.split(aff1$naffairs, SplitRatio = 0.8)# --------- both are X_train , X_test
train <- subset(aff1, split == TRUE)
test <- subset(aff1, split == FALSE)

#----------------------------- Model building on train test evaluation

model_train <- glm(naffairs ~ ., data = train, family = "binomial")  # 0,1
summary(model_train)

# Predict on test data using train_data model
pred_test <- predict(model_train , newdata = test, type = "response")
pred_test


#Confusion matrix
confusion <- table(pred_test > optCutOff, test$naffairs)
confusion

'
         0  1
  FALSE 88 24
  TRUE   2  6
'

# Model Accuracy  on test data

Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 
 #0.7833333


#--------------------------- Compare the model performance on Train data


# Prediction on Train data
pred_train <- predict(model_train, newdata = train, type = "response")
pred_train

#Confusion matrix
confusion <- table(pred_train > optCutOff, train$naffairs)
confusion

'
        0   1
  FALSE 354  99
  TRUE    7  21
'

# Model Accuracy  on test data

Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 
#0.7796258s








