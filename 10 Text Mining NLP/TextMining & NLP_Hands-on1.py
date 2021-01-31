# Text Mining and NLP - Hands-on
#################################

sentence = "We are Learning TextMining from 360DigiTMG"

'TextMining' in sentence # verify if the text is present in the text or not

sentence.index('Learning') # Check the index location

sentence.split().index('TextMining') # Split the sentences into words and present the position

sentence.split()[2] # 3rd word in the sentence 

sentence.split()[2][::-1] # Print the 3rd word in reverse order

words = sentence.split() # All the words in list format

first_word = words[0]

last_word = words[len(words)-1] # Index in the reverse order start with -1

concat_word = first_word + last_word # join 2 words
print(concat_word)

len(words)
oddwords ="War is good ok jk"
# Here words contain String as seperate  "We" "Are" element so while taking i%2 .
#it is taking enitre length of a word
# in odd words .every string character will go in i%2 w,e,a,r,e
[words[i] for i in range(len(words)) if i%2 == 0]
[oddwords[i] for i in range(len(oddwords)) if i%2 == 0] # print the words with even length
# len here it works for index position of elements [ words]
sentence[-3:] # Index in reverse starts from -1

sentence[::-1] # Print entire sentence in reverse order

print(' '.join([word[::-1] for word in words])) # Select each word and print it in reverse 
#'eW era gninraeL'


# Word Tokenization 
import nltk
from nltk import word_tokenize

words = word_tokenize("I am reading NLP Fundamentals") # split() 
print(words)

nltk.pos_tag(words) # Parts of Speech Tagging

nltk.download('stopwords')  # Stop Words from nltk library

from nltk.corpus import stopwords

stop_words = stopwords.words('English') # 179 pre defined stop words
print(stop_words)

sentence1 = "I am learning NLP. It is one of the most popular library in Python"

sentence_words = word_tokenize(sentence1) # Tokenize the sentence
print(sentence_words)

# Filtering stop words from the input string
sentence_no_stops = ' '.join([word for word in sentence_words if word not in stop_words]) 
print(sentence_no_stops)


# Replace words in string
sentence2 = "I visited MY from IND on 14-02-20"

normalized_sentence = sentence2.replace("MY", "Malaysia").replace("IND", "India").replace("-20", "-2020")
print(normalized_sentence)

from autocorrect import Speller # Library to check typos
spell = Speller(lang='en') # supported languages: en, pl, ru, uk, tr, es

spell('Natureal') # Correct spelling is printed

sentence3 = word_tokenize("Ntural Luanguage Processin deals with the art of extracting insightes from Natural Languaes")
print(sentence3)

sentence_corrected = ' '.join([spell(word) for word in sentence3])
print(sentence_corrected)


# Stemming  :Stemming is the process of producing morphological variants of a root/base word
stemmer = nltk.stem.PorterStemmer()

stemmer.stem("Programming")
stemmer.stem("Programs")

stemmer.stem("Jumping")
stemmer.stem("Jumper")

stemmer.stem("battling") # battl - stemming does not look into dictionary words
stemmer.stem("amazing")

# Lemmatization
# Lemmatization looks into dictionary words
#Lemmatization is the process of grouping together the different inflected forms of a word
# so they can be analysed as a single item.
#lemmatization does morphological analysis of the words.
#lemmatize takes a part of speech parameter, “pos” If not supplied, the default is “noun.”
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print("rocks :", lemmatizer.lemmatize("rocks")) 
print("corpora :", lemmatizer.lemmatize("corpora")) 
  
# a denotes adjective in "pos"  better : good
print("better :", lemmatizer.lemmatize("better", pos ="a")) 

lemmatizer.lemmatize('Programming')

lemmatizer.lemmatize('Programs')

lemmatizer.lemmatize('battling')

lemmatizer.lemmatize("amazing")

# Chunking (Shallow Parsing) - Identifying named entities
#The primary usage of chunking is to make a group of "noun phrases." 
#The parts of speech are combined with regular expressions.
nltk.download('maxent_ne_chunker')
nltk.download('words')
sentence4 = "We are learning nlp in Python by 360DigiTMG which is based out of India."


#POS Tagging
#Parts of speech Tagging is responsible for reading the text in a language and assigning some specific token (Parts of Speech) to each word.
nltk.download('averaged_perceptron_tagger')

i = nltk.ne_chunk(nltk.pos_tag(word_tokenize(sentence4)), binary=True)
[a for a in i if len(a)==1]


# Sentence Tokenization
from nltk.tokenize import sent_tokenize
sent_tokenize("We are learning NLP in Python. Delivered by 360DigiTMG. Do you know where is it located? It is based out of India.")


from nltk.wsd import lesk
# Disambiguate .words with different meaning

# How lesk works

#for every sense of the word being disambiguated
# one should count the amount of words that are in both neighborhood of that word 
#and in the dictionary definition of that sense.
#the sense that is to be chosen is the sense which has the biggest number of this count
sentence1 = "Keep your savings in the bank"
print(lesk(word_tokenize(sentence1), 'bank'))

sentence2 = "It's so risky to drive over the banks of the river"
print(lesk(word_tokenize(sentence2), 'bank'))

# "bank" as multiple meanings. 
# The definitions for "bank" can be seen here:
from nltk.corpus import wordnet as wn
for ss in wn.synsets('bank'): print(ss, ss.definition())

#WordNet is the lexical database i.e. dictionary for the English language, 
#specifically designed for natural language processing.

#Synset is a special kind of a simple interface that is present in NLTK to look up words in WordNet.
# Synset instances are the groupings of synonymous words that express the same concept.


syn = wn.synsets('hello')[0] 
  
print ("Synset name :  ", syn.name()) 
  
# Defining the word 
print ("\nSynset meaning : ", syn.definition()) 
  
# list of phrases that use the word in context 
print ("\nSynset example : ", syn.examples()) 

#chunking -- make a chunk or grp or noun 
#######################################
1.	CC	Coordinating conjunction
2.	CD	Cardinal number
3.	DT	Determiner
4.	EX	Existential there
5.	FW	Foreign word
6.	IN	Preposition or subordinating conjunction
7.	JJ	Adjective
8.	JJR	Adjective, comparative
9.	JJS	Adjective, superlative
10.	LS	List item marker
11.	MD	Modal
12.	NN	Noun, singular or mass
13.	NNS	Noun, plural
14.	NNP	Proper noun, singular
15.	NNPS	Proper noun, plural
16.	PDT	Predeterminer
17.	POS	Possessive ending
18.	PRP	Personal pronoun
19.	PRP$	Possessive pronoun
20.	RB	Adverb
21.	RBR	Adverb, comparative
22.	RBS	Adverb, superlative
23.	RP	Particle
24.	SYM	Symbol
25.	TO	to
26.	UH	Interjection
27.	VB	Verb, base form
28.	VBD	Verb, past tense
29.	VBG	Verb, gerund or present participle
30.	VBN	Verb, past participle
31.	VBP	Verb, non-3rd person singular present
32.	VBZ	Verb, 3rd person singular present
33.	WDT	Wh-determiner
34.	WP	Wh-pronoun
35.	WP$	Possessive wh-pronoun
36.	WRB	Wh-adverb
###################################################