
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import matplotlib.pyplot as plt
import seaborn as sns



messages = pd.read_csv("D:\\360Assignments\\Submission\\12 Naive Bayes Classification\\sms_raw_NB.csv",encoding='latin-1')
#EDA
messages.head()
messages.describe()

messages.groupby('type').describe()
'ham 4812 spam 747'


messages['length'] = messages['text'].apply(len)
messages.head()

messages['length'].plot(bins=20, kind='hist') 
#Most messages are of length of between 0 to 200 
# most messages might be just a reply or confirmation near the 0 to 50 mark in histogram
messages.length.describe()
#min         2.000000
#max       910.000000

# viewing the length of text of ham and spam 
messages.hist(column='length', by='type', bins=50,figsize=(12,4))

from nltk.corpus import stopwords
import string
import re

 ############# removing stopwords ,punctuations
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

messages['text'].apply(text_process)

####### removing regular expression 
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))


messages['text'] = messages['text'].apply(cleaning_text)

messages['text'].value_counts()

def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email texts into word count matrix format - Bag of Words
messages_bow = CountVectorizer(analyzer = split_into_words).fit(messages.text)

##################################################### BAG OF WORDS ##############################
# Defining BOW for all messages
messages1 = messages_bow.transform(messages.text)

# For training messages
train_messages = messages_bow.transform(messages.text)

# For testing messages
test_messages = messages_bow.transform(messages.text)

################################################### TFIDF TRANSFORMER ######################################

#TfidfVectorizer is used on sentences,
# while TfidfTransformer is used on an existing count matrix, such as one returned by CountVectorizer

# Learning Term weighting and normalizing on entire emails---------------------TF =
tfidf_transformer = TfidfTransformer().fit(messages1)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_messages)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_messages)
test_tfidf.shape #  (row, column)

###############################################       BAYES STARTS HERE    ####################
# Preparing a naive bayes model on training data set  ------------FIT( TFIDF TRAIN DATA , TRAIN DATA )

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes [ here messages['type'] as on what it will predict .So we give target "COLUMN"]
classifier_mb = MB()
classifier_mb.fit(train_tfidf, messages['type'])

# Evaluation on Test Data-------------------------- PREDICTIONS = MB.PREDICT(TEST DATA)
test_pred_m = classifier_mb.predict(test_tfidf)


#################  MODEL EVALUATION ###################

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(messages['type'],test_pred_m))
'''
              precision    recall  f1-score   support

         ham       0.97      1.00      0.99      4812
        spam       1.00      0.81      0.89       747

    accuracy                           0.97      5559
   macro avg       0.98      0.90      0.94      5559
weighted avg       0.97      0.97      0.97      5559
'''

print(confusion_matrix(messages['type'], test_pred_m))
'''
[[4811    1]
 [ 143  604]]
'''
pd.crosstab(test_pred_m, messages['type']) 
'''
type    ham  spam
row_0            
ham    4811   143
spam      1   604
''' 
