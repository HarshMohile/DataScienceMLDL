library(recommenderlab)
library(reshape2)
library(readxl)
library(ggplot2)

joke_rating <- read_excel(file.choose())
head(joke_rating)
joker <- joke_rating[2:4]
head(joker)
dim(joker)
class(joker)

joke_matrix <- as.matrix(acast(joker, user_id~joke_id, fun.aggregate = mean))
dim(joke_matrix)
hist(joke_matrix)

?getRatings
hist(getRatings(joke_matrix))


R <- as(joke_matrix, "realRatingMatrix")

rec1 = Recommender(R, method="UBCF")
rec1

userid=2898

jokes_final <- subset(joker, joker$user_id==userid)
print("You have rated these jokes :")
jokes_final



print("recommendations for you:")
prediction <- predict(rec1, R[userid], n=5) ## n= no of recommendation 
as(prediction, "list")



min(joke_rating$Rating)
min(joke_rating)


class(joke_rating)

# user with maxnimum rating given as 10
joke_rating[which.max(joke_rating$Rating),]

#user with lowest rating given for joke
joke_rating[which.min(joke_rating$Rating),]


#Recommending  a joke for the lowest rat given user by UBCF.
userid=18502

jokes_final <- subset(joker, joker$user_id==userid)
print("You have rated these jokes :")
jokes_final



print("recommendations for you:")
prediction <- predict(rec1, R[userid], n=5) ## n= no of recommendation 
as(prediction, "list")

  