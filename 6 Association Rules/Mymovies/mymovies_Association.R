library("arules")
library("arulesViz")
library("readr")

#movies <- read.transactions("D:\\360Assignments\\Submission\\Association Rules 6\\my_movies.csv",sep = ",")
movies <- read.csv(file.choose())
class(movies)

summary(movies)

#binary matrix 
movies_transaction = as.matrix(movies[6:15])
class(movies_transaction)

View(movies_transaction)

mrules <- apriori(movies_transaction,parameter=list(support=0.2, confidence = 0.5,minlen=2))
plot(mrules,method="grouped")

summary(mrules)
inspect(sort(mrules, by = "lift"))

plot(mrules,method="graph")

