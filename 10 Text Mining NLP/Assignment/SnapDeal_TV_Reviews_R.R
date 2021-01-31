
library(twitteR)
library(ROAuth)
library(base64enc)
library(httpuv)
library(rvest)
library(XML)
library(magrittr)
library(tm)

#full ur is formed by concat of  paste(url , pagenumber is i , seperated by =) . 
aurl <- "https://www.snapdeal.com/product/lg-42lb5510-42-inches-fully/532808444/reviews?page"

LG_reviews <- NULL

for (i in 1:3){
  surl <- read_html(as.character(paste(aurl,i,sep="=")))
  rev <- surl %>% html_nodes(".user-review") %>% html_text()
  LG_reviews <- c(LG_reviews,rev)
}

??html_nodes
??html_text



write.table(LG_reviews,"Snapdeal_LGTV.txt")
getwd()


# backing up our LG review in txt variable
txt <- LG_reviews
str(txt)
length(txt)
View(txt)


# Convert the character data to corpus type
x <- Corpus(VectorSource(txt))
?VectorSource

inspect(x[1]) #Metadata:  corpus specific: 1, document level (indexed): 0
#Content:  documents: 1

#returns all 36 reviews
for (j in 1:36){
  inspect(x[j])  
}

 ?tm_map
# tm_map takes corpus (coll of docs) and applies/transforms the function (mean, sum , remove punctuation)
#on each and every document in corpus.
x1 <- tm_map(x, tolower)
inspect(x1[1])

x1 <- tm_map(x1, removePunctuation)
inspect(x1[1])

inspect(x1[5])
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
#Maximal term length: 17
#Sparsity : 94%
dtm <- t(tdm) # transpose
dtm <- DocumentTermMatrix(x1)

tdm[["dimnames"]][["Docs"]] # to see the  number fo docs is number of reviews (ie.36)


tdm1 <- removeSparseTerms(tdm, 0.94) 
?removeSparseTerms

tdm1 <- as.matrix(tdm1)
dim(tdm)
#before removesparse dim>>278  36
dim(tdm1)

#viewing the matrix of TDM of 20 terms(rows) and 20 docs(cols)
tdm1[1:20, 1:20]

#good           1 2 0 1 2 1 0 2 1  2  0  0  1  2  0  1  0  1  1  0 = 40
#each rowvalue is a [doc] in which "good" is present and  its frequency in each doc
#.(below we are summing it up)


# Bar plot:  [grouping(sum) by and giving the count . i.e its frequency]
w <- rowSums(tdm1)
w
# bought          first           gone           good gooooooooooood        helpful          lifes 
#   5              8              4             40              4              6              4 
#basically bag of words(vector) with its frequency but output in this is not a vector
w_sub <- subset(w, w >= 20)
w_sub

#  good  product    buyer verified 
#40       41       30       30
# above the words"product" "buyer" "verified" are of no use.


barplot(w_sub, las=2, col = rainbow(30))

# Terms repeats maximum number of times
# we cannot directly put tdm in barplot we have used x1 (corpusvector) .
#then data cleaning and then converting it into tdm again and then into w_sub=subset(w,w>=30)
#1.corups into tdm ( tdm as matrix)
#2.summing the rows w= rowSums(tdm)
#3.subset and then barplot of subset of w>=40

x2 <- tm_map(x1, removeWords, c('product','buyer','verified'))
x2 <- tm_map(x1, stripWhitespace)

tdm <- TermDocumentMatrix(x2)
tdm
class(tdm)
tdm <- as.matrix(tdm)
tdm[100:109, 1:20]

# Bar plot after removal of the term 'product' , 'buyer','verified'
w <- rowSums(tdm)
w

w_sub <- subset(w, w >= 10)
w_sub

barplot(w_sub, las=2, col = rainbow(30))

#  good snapdeal delivery  service    worst  quality 
#40       15       16       12       12       14

##### Word cloud ##### Data Visualization #########
install.packages("wordcloud")
library(wordcloud)

?wordcloud
wordcloud(words = names(w_sub), freq = w_sub)

# no conditon >= [all are considered]
w_sub1 <- sort(rowSums(tdm), decreasing = TRUE)
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
w_df <- data.frame(names(w_sub), w_sub)

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
bitoken <- NGramTokenizer(x1, Weka_control(min = 2, max = 2))
class(bitoken) # character
# ?table: the counts at each combination of factor levels. |word |freq |
two_word <- data.frame(table(bitoken))
# slicing df[df[""Freq"]]to get the highest frequency of bigram
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


pos.matches[pos.matches=='TRUE']
pos.matches[pos.matches=='FALSE']

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

