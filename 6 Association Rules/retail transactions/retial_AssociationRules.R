library("arules")
library("arulesViz")
library("readr")

retail <- read.csv(file.choose())
class(retail)

retail_transactions <- as(retail,"transactions")
class(retail_transactions)

rrules <- apriori(retail_transactions,parameter=list(support=0.007, confidence = 0.005,minlen=2)) 
plot(rrules)

rrules1 <- apriori(retail_transactions,parameter=list(support=0.003, confidence = 0.8,minlen=5)) 
plot(rrules1)

inspect(head(sort(rrules1, by = "lift")))

plot(rrules1,method="graph")
plot(rrules1,method="grouped")
