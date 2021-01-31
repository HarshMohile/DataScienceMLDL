library(plyr)
library(animation)
library(readr)
library(cluster)

wine <- read_csv(file.choose())
normalized_data <- scale(wine)

# find TWss for for each cluster , here TWSS will be no.of columns
twss <- NULL
for (i in 1:14) {
  twss <- c(twss, kmeans(normalized_data, centers = i)$tot.withinss)
}
twss

plot(1:14, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")
# k =3 
# Animated cluster diagram for k=3
fit <- kmeans.ani(normalized_data, 3) #trying with 3 clusters to get minimum withinss.
str(fit)
fit$centers # gives final centroid of each clusters
fit$cluster
View(final <- data.frame(fit$cluster, wine))

View(aggregate(wine, by = list(fit$cluster), FUN = mean)) #group by cluster (1,2,3) find mean

# Actual cluster diagram for k=3
clust <- clara(normalized_data,3)
clusplot(clust)
