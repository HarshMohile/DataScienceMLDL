library("arules")
library("arulesViz")
library("readr")

books <- read.csv(file.choose())
class(books)



book_transaction = as.matrix(books)

summary(book_transaction)

brules <- apriori(book_transaction,parameter=list(support=0.02, confidence = 0.5,minlen=5))

plot(brules)



plot(brules, method="grouped") 
plot(brules ,method="graph")

rules_conf <- sort (brules, by="lift", decreasing=TRUE)
# 'high-confidence' rules.

inspect(head(rules_conf)) # consider higher lift value.rather than relying on conf to find frequent items

plot(rules_conf)

