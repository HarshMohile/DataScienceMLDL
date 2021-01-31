library(plyr)
library(animation)
library(readr)
library(cluster)

wine <- read_csv(file.choose())

?princomp
# correlation matrix , calc the PCA score of each component and covariance matrix is set to NULL
pcaObj <- princomp(wine, cor = TRUE, scores = TRUE, covmat = NULL)

str(pcaObj)
summary(pcaObj)

loadings(pcaObj)


# the Std.Dev keeps reducing as more and more components are set


plot(pcaObj) # graph showing importance of principal components
biplot(pcaObj)

?cumsum
# cumsum(pcaObj$sdev * pcaObj$sdev) * 100   gives higher values as component goes on 
#Comp1 = 553.5948   Comp12= 1381.2624
# High variance on earlier component and have lots of dynamic data
# plot to see which n-components should we have ,here its 3

plot(cumsum(pcaObj$sdev * pcaObj$sdev) * 100 / (sum(pcaObj$sdev * pcaObj$sdev)), type = "b")

pcaObj$scores
dim(pcaObj$scores) # 178 *14
# Top 3 pca scores [first 3 Cols , all 178 rows]
pcaObj$scores[, 1:3]

final <- cbind(wine[, 1], pcaObj$scores[, 1:3])
View(final)

class(final)
# Scatter diagram
plot(final[,1], final[,2])


###################### Kmeans After applying PCA###############################

class(final)
wine_df <-  as.data.frame(final[,-1])


# find TWss for for each cluster , here TWSS will be no.of columns
twss <- NULL
for (i in 1:3) {
  twss <- c(twss, kmeans(wine_df, centers = i)$tot.withinss)
}
twss

plot(1:3, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")

# k =2 
# Animated cluster diagram for k=2
fit <- kmeans.ani(wine_df, 2) #trying with 2 clusters to get minimum withinss.
str(fit)
fit$centers # gives final centroid of each clusters
fit$cluster
View(final <- data.frame(fit$cluster, wine))

View(aggregate(wine, by = list(fit$cluster), FUN = mean)) #group by cluster (1,2,3) find mean

# Actual cluster diagram for k=2
clust <- clara(wine_df,2)
clusplot(clust)
# These 2 components explain 66.67% of the point variability.