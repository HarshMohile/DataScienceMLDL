library(plyr)
library(animation)
library(readxl)
library(readr)

crime <- read_csv(file.choose())
crime <- crime[, c(-1)]
plot(crime[])

normalized_data <- scale(crime)

twss <- NULL
for (i in 1:4) {
  twss <- c(twss, kmeans(normalized_data, centers = i)$tot.withinss)
}
twss


plot(1:4, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")

# elbow curve noticed on 2.0 number of cluster and we can keep n=2 for finding  kmean cluster

fit <- kmeans.ani(normalized_data, 2) #trying with 2 clusters to get minimum withinss.
str(fit)
fit$cluster
fit$centers


final <- data.frame(fit$cluster, crime)

aggregate(crime, by = list(fit$cluster), FUN = mean)