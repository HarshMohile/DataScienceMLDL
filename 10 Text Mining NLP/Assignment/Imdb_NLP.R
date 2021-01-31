
library(twitteR)
library(ROAuth)
library(base64enc)
library(httpuv)
library(rvest)
library(XML)
library(magrittr)
library(tm)

#full ur is formed by concat of  paste(url , pagenumber is i , seperated by =) . 
jurl <- "https://www.imdb.com/title/tt7286456/reviews"

imdb_reviews <- NULL

  url <- read_html(as.character(jurl))
  rev <- url %>% html_nodes(".lister-list") %>% html_text()
  imdb_reviews <- c(imdb_reviews,rev)


??html_nodes
??html_text



write.table(imdb_reviews,"Joker_imdb1.txt")
getwd()


# backing up our imdb review in txt variable
txt <- imdb_reviews
str(txt)
length(txt)
View(txt)


# Convert the character data to corpus type
x <- Corpus(VectorSource(txt))
?VectorSource

inspect(x[1]) #Metadata:  corpus specific: 1, document level (indexed): 0
#Content:  documents: 1




?tm_map
# tm_map takes corpus (coll of docs) and applies/transforms the function (mean, sum , remove punctuation)
#on each and every document in corpus.
x1 <- tm_map(x, tolower)
inspect(x1[1])

x1 <- tm_map(x1, removePunctuation)
inspect(x1[1])


x1 <- tm_map(x1, removeNumbers)
inspect(x1[1])

x1 <- tm_map(x1, removeWords, stopwords('english'))
inspect(x1[1])

# striping white spaces 
x1 <- tm_map(x1, stripWhitespace)
inspect(x1[1])



# Term document matrix 
# converting unstructured data to structured format using TDM

# converting corpus(collection of docs) into Doc-term matrix to give its frequency. 
?TermDocumentMatrix
tdm <- TermDocumentMatrix(x1)
tdm
#Maximal term length: 35
#Sparsity : 0% [ 1  document ]
dtm <- t(tdm) # transpose
dtm <- DocumentTermMatrix(x1)

tdm[["dimnames"]][["Docs"]] # to see the  number fo docs is number of reviews


tdm <- as.matrix(tdm)
dim(tdm)
#[1] 950   1


#each rowvalue is a [doc] in which "good" is present and  its frequency in each doc
#.(below we are summing it up)


# Bar plot:  [grouping(sum) by and giving the count . i.e its frequency]
w <- rowSums(tdm)
w

#able                          absolutely 
#1                                   3 
#bsurd                        accomplishes 
#1                                   1 

#basically bag of words(vector) with its frequency but output in this is not a vector
w_sub <- subset(w, w >= 30)
w_sub

#film     found   helpful     joker     movie permalink    review      sign      vote 
#24        26        50        30        67        25        29        27        25 
# above the word [film, found mhelpful,movie, permalink,review ,sign ,vote,movie] are of no use.


barplot(w_sub, las=2, col = rainbow(30))

# Terms repeats maximum number of times
# we cannot directly put tdm in barplot we have used x1 (corpusvector) .
#then data cleaning and then converting it into tdm again and then into w_sub=subset(w,w>=30)
#1.corups into tdm ( tdm as matrix)
#2.summing the rows w= rowSums(tdm)
#3.subset and then barplot of subset of w>=40

x2 <- tm_map(x1, stripWhitespace)
x2 <- tm_map(x1, removeWords, c('film', 'found' ,'helpful','movie', 
                                'permalink','review' ,'sign' ,'vote','joker',
                                'also','just','like','people','say','seen','think',
                                'time','will','get','one','really'))


tdm1 <- TermDocumentMatrix(x2)
tdm1
class(tdm1)
tdm1 <- as.matrix(tdm1)
tdm1

# Bar plot after removal of the term 'phone'
w <- rowSums(tdm1)
w

w_sub <- subset(w, w >= 10)
w_sub

barplot(w_sub, las=2, col = rainbow(30))

#acting      best character      good   joaquin   october   phoenix      plot   society 
#12        16        10        15        17        18        16        12        12 

##### Word cloud ##### Data Visualization #########
install.packages("wordcloud")
library(wordcloud)

?wordcloud
wordcloud(words = names(w_sub), freq = w_sub)

# no conditon >= [all are considered]  in highest to lowest
w_sub1 <- sort(rowSums(tdm1), decreasing = TRUE)
head(w_sub1)

wordcloud(words = names(w_sub1), freq = w_sub1) # all words are considered

# better visualization
wordcloud(words = names(w_sub1), freq = w_sub1, random.order=F, colors=rainbow(30), scale = c(2,0.5), rot.per = 0.4)
windows()
wordcloud(words = names(w_sub1), freq = w_sub1, random.order=F, colors= rainbow(30),scale=c(3,0.5),rot.per=0.3)
?wordcloud

windowsFonts(JP1 = windowsFont("MS Gothic"))
par(family = "JP1")
wordcloud(x1, scale= c(2,0.5))
?windowsFonts

############# Wordcloud2 ###############requires input as dataframe

install.packages("wordcloud2")
library(wordcloud2)
?data.frame
w_df <- data.frame(names(w_sub), w_sub) #creates 2 cols names(w_sub) and  w_sub(i.e frequency)

colnames(w_df) <- c('word', 'freq')

wordcloud2(w_df, size=0.3, shape='circle')
?wordcloud2

wordcloud2(w_df, size=0.3, shape = 'triangle')
wordcloud2(w_df, size=0.3, shape = 'star')


#### Bigram ####
library(RWeka)
library(wordcloud)

??RWeka
??NGramTokenizer
# NgramTokenizer ( input vector to be tokenized , Wekacontrol( for more controlfor ngrams))
??Weka_control
minfreq_bigram <- 2
# only takes corpus(x1) as an input .Not a tdm 
bitoken <- NGramTokenizer(x2, Weka_control(min = 2, max = 2))
class(bitoken) # character

# ?table: the counts at each combination of factor levels. |word |freq |.
#dataframe off the table() becuase we want it in row and column
two_word <- data.frame(table(bitoken))

# slicing df[df[""Freq"]]to get the highest frequency of bigram descending order.
#order instead of sort. sort in this case returns every bigram as 1 freq 
#sort() sorts the vector in an ascending order.
#order() returns the indexes[] of the vector in a sorted order.

sort_two <- two_word[order(two_word$Freq, decreasing = TRUE), ]

wordcloud(sort_two$bitoken, sort_two$Freq, random.order = F, 
          scale = c(2, 0.35), min.freq = minfreq_bigram,
          colors = brewer.pal(8, "Dark2"), max.words = 150)

#####################################
# lOADING Positive and Negative words  
pos.words <- readLines(file.choose())	# read-in positive-words.txt
neg.words <- readLines(file.choose()) 	# read-in negative-words.txt

stopwdrds <-  readLines(file.choose())

### Positive word cloud ###
pos.matches <- match(names(w_sub1), pos.words)
pos.matches <- !is.na(pos.matches)


sum(pos.matches[pos.matches=='TRUE']) # 84 positive words are found matching
sum(pos.matches[pos.matches=='FALSE']) # 0 because others NA|False are removed

class(pos.matches)

# getting only those WORDS that are present in w_sub1 and pos.txt and its frequency
# bag of words of only positive  words
freq_pos <- w_sub1[pos.matches]
names <- names(freq_pos)
windows()
wordcloud(names, freq_pos, scale=c(4,1), colors = brewer.pal(8,"Dark2"))


### Matching Negative words ###

neg.matches <- match(names(w_sub1), neg.words)
neg.matches <- !is.na(neg.matches)
freq_neg <- w_sub1[neg.matches]
#names <- names(freq_neg)
windows()
wordcloud(names(freq_neg), freq_neg, scale=c(4,.5), colors = brewer.pal(8, "Dark2"))

