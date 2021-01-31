library(plyr)
library(animation)
library(readxl)
install.packages("cluster")
library(cluster)

airline <- read_excel(file.choose())
airline <- airline[,c(-1)]

normalized_data <- scale(airline)



twss <- NULL
for (i in 1:11) {
  twss <- c(twss, kmeans(normalized_data, centers = i)$tot.withinss)
}
twss

plot(1:11, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")


fit <- kmeans.ani(normalized_data, 4) #trying with 4 clusters to get minimum withinss.
str(fit)
fit$centers # gives final centroid of each clusters
fit$cluster
View(final <- data.frame(fit$cluster, airline))

View(aggregate(airline, by = list(fit$cluster), FUN = mean)) #to match cols with their resp. mean and cluster number for inference

clust <- clara(normalized_data,4)
clusplot(clust)
