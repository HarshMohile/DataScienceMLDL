import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 

import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

imdb_re=[]


ip=[]  
url="https://www.imdb.com/title/tt7286456/reviews"
response = requests.get(url)
soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
reviews = soup.find_all("div",attrs={"class","lister-list"})# Extracting the content under specific tags  
for i in range(len(reviews)):
    ip.append(reviews[i].text)  
 
    imdb_re=imdb_re+ip  # adding the reviews of one page to empty list which in future contains all the reviews

imdb_re

os.getcwd()
with open("Joker_imdb.txt","w",encoding='utf8') as output:
    output.write(str(imdb_re))
    
    
# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(imdb_re)
ip_rev_string

# Removing unwanted symbols incase if exists (Regular expressions are substituted by " ")
ip_rev_string = re.sub("[^A-Za-z" "]+"," ", ip_rev_string).lower()
ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string)
ip_rev_string


ip_reviews_words = ip_rev_string.split(" ")

#Tokenized
ip_reviews_words

#TFIDF :Convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.feature_extraction.text import TfidfVectorizer

# creating a object of TfidfVectorizer and fitting and transfroming our data
vectorizer = TfidfVectorizer(ip_reviews_words, use_idf=True,ngram_range=(1, 3))
X = vectorizer.fit_transform(ip_reviews_words)

#Importing Stopwords from NLTK
from nltk.corpus import stopwords

# Taking stop words from STOPWORDS method into a variable so that later it can be filtererd for our test data
stop_words = stopwords.words('English') 
print(stop_words)
# Filtering stop words from the input string
ip_reviews_words= ' '.join([word for word in ip_reviews_words if word not in stop_words]) 
print(ip_reviews_words)


wordcloud_ip = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_reviews_words)

plt.imshow(wordcloud_ip)

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

other_stopwords_to_remove = ['film','joker','helpful','review','found','movie','vote','permalink','sign vote','time','sign']
STOPWORDS = STOPWORDS.union(set(other_stopwords_to_remove))
stopwords = set(STOPWORDS)

wordcloud = WordCloud(width = 1800, height = 1800, 
                background_color ='white', 
                max_words=200,
                stopwords = stopwords, 
                min_font_size = 10).generate(ip_reviews_words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.figure(figsize=(12,7))

import pandas as pd
import numpy as np

os.getcwd()
with open("Joker_review.csv","w",encoding='utf8') as output:
    output.write(str(ip_reviews_words))
    
imdb_df= pd.read_csv("Joker_review.csv",names=['Reviews'])

imdb_df.head()

from textblob import TextBlob

imdb_df['number_of_words'] = imdb_df['Reviews'].apply(lambda x : len(TextBlob(str(x)).words))
imdb_df['number_of_words']

# Polarity
imdb_df['polarity'] = imdb_df['Reviews'].apply(lambda x : TextBlob(str(x)).sentiment.polarity)
imdb_df['polarity']

# Subjectivity
imdb_df['subjectivity'] = imdb_df['Reviews'].apply(lambda x : TextBlob(str(x)).sentiment.subjectivity)
imdb_df['subjectivity']

# BAG OF WORDS
from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer().fit(imdb_df['Reviews'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))
bag_of_word_df = pd.DataFrame(bow_transformer.fit_transform(imdb_df['Reviews']).todense())
bag_of_word_df

bag_of_word_df.columns = sorted(bow_transformer.vocabulary_)
bag_of_word_df.head()



### ALGORITHM of NLP
'''
take data from internet usig Beautifulsoup ,request, response, 

Req the url and put it in response
from response get content and put in it soup  by html parser
 reviews =soup.find all by xpath( div ,classname)
  iterater over the review bty using for loopand append to other array lol.
  
Split the reviews into singular unit 

Use TFIDF vectorizer and then remove stopwords 
 WordCloud
 
 
 use Textblob to learn about Sentiments ,polarity , subjectiviity
 
 
 USe CountVectorizer to form Bag of words by fit
 
 then print the total number of vocab words 
 len(bow_transformer.vocabulary_)

'''




