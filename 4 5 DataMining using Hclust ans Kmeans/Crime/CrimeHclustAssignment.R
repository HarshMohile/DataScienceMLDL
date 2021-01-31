libaray(readr)
crime <- read.csv(file.choose())
mycrime <- crime[,c(-1)]

normalized_data <- scale(mycrime)

summary(mycrime)

d <- dist(normalized_data, method = "euclidean") 

fit <- hclust(d, method = "complete")
plot(fit,hang = -1)

groups <- cutree(fit, k = 3)

rect.hclust(fit, k = 3, border = "red")

membership <- as.matrix(groups)

final <- data.frame(membership, crime)

aggregate(mycrime, by = list(final$membership), FUN = mean)

library(readr)
write_csv(final, "hclustoutputcrime.csv")

getwd()

