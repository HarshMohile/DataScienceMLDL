library("arules")
library("arulesViz")
library("readr")

groceries <- read.transactions("D:\\360Assignments\\Submission\\Association Rules 6\\groceries.csv",sep = ",")

summary(groceries)
class(groceries)

#most frequent items:
#  whole milk other vegetables       rolls/buns             soda           yogurt          (Other) 
#2513             1903             1809             1715             1372            34055 


inspect(head(groceries[1:5]))


grules <- apriori(groceries,parameter=list(support=0.008, confidence = 0.05,minlen=2))
grules1 <- apriori(groceries,parameter=list(support=0.003, confidence = 0.7,minlen=2))
plot(grules)

summary(grules)

inspect(sort(grules, by = "lift"))
#likely brought item berrie along with whipped/sour cream

inspect(sort(grules, by = "support"))[1:5]

plot(grules,method="grouped")
plot(grules,method="graph")
