library(readr)
library(animation)

insr <- read.csv(file.choose())

nor_insr <- scale(insr,center = TRUE)

fit <- kmeans.ani(nor_insr,3)
fit$cluster
fit$centers
View(str(fit))
summary(fit)
View(final <- data.frame(fit$cluster, insr)) # majority belongs to cluster 3 before elbow curve plot

twss <- NULL
for (i in 1:5) {
  twss <- c(twss, kmeans(normalized_data, centers = i)$tot.withinss)
}
twss

plot(1:5, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")


fit <- kmeans(nor_insr,2)
fit$cluster
fit$centers
str(fit)
summary(fit)
View(final <- data.frame(fit$cluster, insr)) # majority belongs to cluster 2
#totalss - totwithinss = between 
# 495  -   314 == 181


aggregate(insr, by = list(fit$cluster), FUN = mean)

