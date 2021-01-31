# Logistic regeression On bank data
library(readr)

bank <- read.csv(file.choose())

# Data cleaning 
sum(is.na(bank))

# Omitting NA values from the Data 
bank <- na.omit(bank) # na.omit => will omit the rows which has atleast 1 NA value
dim(bank)


# See the proportion 
prop.table(table(bank$y))


##0.8830152 0.1169848 for 0 and 1  resp.

names(bank)

cor(bank$age, bank$balance) # low corrleation 
cor(bank) # correlation matrix





#------------------------------------ model building initial using linear regresssion 
# Preparing a linear regression 
mod_lm <- lm(y ~ ., data = bank)
summary(mod_lm)

# joself.employed ,joblue.collar ,joself.employed ,joentrepreneur 
#have high p values which can be removed  Since they dont add any significance to predict y 

pred1 <- predict(mod_lm, bank) # whole bank_data as test 
pred1

#plot(bank$y, pred1)

#------------enitre dataset------------- Model building using GLM model for LOGistic rgeresssion

model <- glm(y ~ ., data = bank, family = "binomial")  # 0,1
summary(model)


# for taking  Log Ratio first take exp() of  coef(model)
exp(coef(model))

# there are NA in coef(Model)

# Prediction
pred_prob <- predict(model, bank, type = "response")
pred_prob


# threshold on when to classify Yes or No , 0 or 1
confusion <- table(pred_prob > 0.5, bank$y)
confusion

'
            0     1
  FALSE 39013  3587
  TRUE    909  1702
'

# Model Accuracy 
Acc <- sum(diag(confusion)/sum(confusion))
Acc  
#0.90

# Convert the probabilities to binary output form using cutoff
pred_values <- ifelse(pred_prob > 0.5, 1, 0)

library(caret)
# Confusion Matrix in a detailed format
# confusionMatrix(actual , predicted) but in factor( levels=c(0,1))

confusionMatrix(factor(bank$y, levels = c(0, 1)), factor(pred_values, levels = c(0, 1)))


# Decide on optimal prediction probability cutoff for the model
library(InformationValue)
optCutOff <- optimalCutoff(bank$y, pred_prob)
optCutOff
#  0.4699999

# Check multicollinearity in the model using VIF
library(car) 
vif(model)

#  There are aliased coefficients (NA) in the model so we check the culprits
#the linearly dependent variables

#find the aliased cols
ld.vars <- attributes(alias(model)$Complete)$dimnames[[1]]

#"poutunknown" "con_unknown" "single"      "jounknown"  .
#Remove these colS, [SUBSET FOR DATA]
bank1 <-  subset(bank,select =-c(poutunknown,con_unknown,single,jounknown))


model <- glm(y ~ ., data = bank1, family = "binomial")  # 0,1
summary(model)

# Check multicollinearity in the model using VIF
library(car) 
vif_model <-  vif(model) # multicoliinearity detected


##**** Showing the vars with high COllinearity 
'
for( i in  names(vif_model)){
   vi_model <- list(vif_model[i]>10)
   print(vi_model)
}
'
library(dplyr)

vif_df <- as.data.frame(vif_model)

vif_df %>%
  filter_all(all_vars(. > 10))

#******  
#*
#*Removing those with high VIF(Collin) as well
bank2 <-  subset(bank1,select =-c(joadmin.,joblue.collar,jomanagement,joretired,joservices,jotechnician))

# Check multicollinearity in the model using VIF again (1)

model <- glm(y ~ ., data = bank2, family = "binomial")  # 0,1
summary(model)

library(car) 
vif_model <-  vif(model) # Collinearity Resovled


prob_full <- predict(model, bank2, type = "response")
prob_full

# Decide on optimal prediction probability cutoff for the model
library(InformationValue)
optCutOff <- optimalCutoff(bank2$y, prob_full)
optCutOff
#  0.4599999

# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(bank2$y, prob_full, threshold = optCutOff)

#0.0992

# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(bank2$y, prob_full)

# ROC curve with AUROC  gave about 0.88

# Confusion Matrix
predvalues <- ifelse(prob_full > optCutOff, 1, 0)

results <- confusionMatrix(predvalues, bank2$y)

'
      0    1
0 38861 1061
1  3426 1863
'

sensitivity(predvalues, bank2$y)   
confusionMatrix(actuals = bank2$y, predictedScores = predvalues)


####----- -----------------------Splitting the dataset -------------------

library(caTools)
set.seed(0)
split <- sample.split(bank2$y, SplitRatio = 0.8)# --------- both are X_train , X_test
train <- subset(bank2, split == TRUE)
test <- subset(bank2, split == FALSE)

#----------------------------- Model building on train test evaluation

model_train <- glm(y ~ ., data = train, family = "binomial")  # 0,1
summary(model)

# Predict on test data using train_data model
pred_test <- predict(model_train , newdata = test, type = "response")
pred_test

#Confusion matrix
confusion <- table(pred_test > optCutOff, bank2$y)
confusion

'
           0     1
  FALSE 39013  3587
  TRUE    909  1702
'

# Model Accuracy  on test data

Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 

#--------------------------- Compare the model performance on Train data


# Prediction on Train data
pred_train <- predict(model_train, newdata = train, type = "response")
pred_train






