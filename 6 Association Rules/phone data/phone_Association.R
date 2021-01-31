library("arules")
library("arulesViz")
library("readr")

#movies <- read.transactions("D:\\360Assignments\\Submission\\Association Rules 6\\my_movies.csv",sep = ",")
phonedata <- read.csv(file.choose())
class(phonedata)

summary(phonedata)

#binary matrix 
phone_transaction = as.matrix(phonedata[4:9])
class(phone_transaction)

View(phone_transaction)

prules <- apriori(phone_transaction,parameter=list(support=0.03, confidence = 0.5,minlen=2)) # 6 rules
plot(prules,method="grouped")

summary(prules)
inspect(sort(prules, by = "support"))
inspect(sort(prules, by = "lift"))

plot(prules,method="graph")


