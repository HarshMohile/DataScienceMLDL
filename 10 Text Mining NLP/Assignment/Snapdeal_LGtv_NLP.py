import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 

import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# creating empty reviews list 
LG_reviews=[]
for i in range(1,4):
  ip=[]  
  url="https://www.snapdeal.com/product/lg-42lb5510-42-inches-fully/532808444/reviews?page="+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.find_all("div",attrs={"class","user-review"})# Extracting the content under specific tags  
  for i in range(len(reviews)):
    ip.append(reviews[i].text)  
 
  LG_reviews=LG_reviews+ip  # adding the reviews of one page to empty list which in future contains all the reviews

LG_reviews

os.getcwd()
with open("LGReviews.txt","w",encoding='utf8') as output:
    output.write(str(LG_reviews))
    
        
# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(LG_reviews)
ip_rev_string

# Removing unwanted symbols incase if exists (Regular expressions are substituted by " ")
ip_rev_string = re.sub("[^A-Za-z" "]+"," ", ip_rev_string).lower()
ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string)
ip_rev_string

# words that contained in LG TV reviews
ip_reviews_words = ip_rev_string.split(" ")

#Tokenized
ip_reviews_words


#** TF-IDF is a statistical measure that evaluates how relevant a word is to a document in a collection of documents. This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents.**
#TFIDF :Convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.feature_extraction.text import TfidfVectorizer


# creating a object of TfidfVectorizer and fitting and transfroming our data
vectorizer = TfidfVectorizer(ip_reviews_words, use_idf=True,ngram_range=(1, 3))
X = vectorizer.fit_transform(ip_reviews_words)

#Importing Stopwords from NLTK
from nltk.corpus import stopwords
X.shape

# Taking stop words from STOPWORDS method into a variable so that later it can be filtererd for our test data
stop_words = stopwords.words('English') 
print(stop_words)

# Filtering stop words from the input string
ip_reviews_words= ' '.join([word for word in ip_reviews_words if word not in stop_words]) 
print(ip_reviews_words)
# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)

wordcloud_ip = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)

plt.imshow(wordcloud_ip)

ip_rev_string

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

other_stopwords_to_remove = ['product', 'verified buyer','may verified','verified','buyer','snapdeal','lg','tv']
STOPWORDS = STOPWORDS.union(set(other_stopwords_to_remove))
stopwords = set(STOPWORDS)

wordcloud = WordCloud(width = 1800, height = 1800, 
                background_color ='white', 
                max_words=200,
                stopwords = stopwords, 
                min_font_size = 10).generate(ip_rev_string)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.figure(figsize=(12,7))

import pandas as pd
import numpy as np

os.getcwd()
with open("LG_Semantic.csv","w",encoding='utf8') as output:
    output.write(str(ip_rev_string))
    
lg_df= pd.read_csv("LG_Semantic.csv",names=['Reviews'])

from textblob import TextBlob

lg_df['number_of_words'] = lg_df['Reviews'].apply(lambda x : len(TextBlob(str(x)).words))
lg_df['number_of_words']

# Detect presence of wh words
wh_words = set(['why', 'who', 'which', 'what', 'where', 'when', 'how'])
lg_df['is_wh_words_present'] = lg_df['Reviews'].apply(lambda x : True if len(set(TextBlob(str(x)).words).intersection(wh_words))>0 else False)
lg_df['is_wh_words_present']

# Polarity
lg_df['polarity'] = lg_df['Reviews'].apply(lambda x : TextBlob(str(x)).sentiment.polarity)
lg_df['polarity']

# Subjectivity
lg_df['subjectivity'] = lg_df['Reviews'].apply(lambda x : TextBlob(str(x)).sentiment.subjectivity)
lg_df['subjectivity']

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer().fit(lg_df['Reviews'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))
bag_of_word_df = pd.DataFrame(bow_transformer.fit_transform(lg_df['Reviews']).todense())
bag_of_word_df

bag_of_word_df.columns = sorted(bow_transformer.vocabulary_)
bag_of_word_df.head()