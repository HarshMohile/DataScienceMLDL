library(recommenderlab)
library(reshape2)

####### Example: Data generated in class #####
ratings_list <-read.csv("Move ratings.csv",header=TRUE)
head(ratings_list)
ratings_list <- ratings_list[,2:4]
head(ratings_list)
dim(ratings_list)

## covert to matrix format
?acast
ratings_matrix <- as.matrix(acast(ratings_list, user~movie, fun.aggregate = mean))
dim(ratings_matrix)

## recommendarlab realRatingMatrix format
R <- as(ratings_matrix, "realRatingMatrix")

rec1 = Recommender(R, method="UBCF") ## User-based collaborative filtering
rec2 = Recommender(R, method="IBCF") ## Item-based collaborative filtering
rec3 = Recommender(R, method="SVD")
rec4 = Recommender(R, method="POPULAR")
rec5 = Recommender(binarize(R,minRating=2), method="UBCF") ## binarize all 2+ rating to 1

## create n recommendations for a user
uid = "Ravi"
movies <- subset(ratings_list, ratings_list$user==uid)
print("You have rated:")
movies
print("recommendations for you:")
prediction <- predict(rec1, R[uid], n=2) ## you may change the model here
as(prediction, "list")
prediction <- predict(rec2, R[uid], n=2) ## you may change the model here
as(prediction, "list")
prediction <- predict(rec3, R[uid], n=2) ## you may change the model here
as(prediction, "list")
prediction <- predict(rec4, R[uid], n=2) ## you may change the model here
as(prediction, "list")
prediction <- predict(rec5, R[uid], n=2) ## you may change the model here
as(prediction, "list")
