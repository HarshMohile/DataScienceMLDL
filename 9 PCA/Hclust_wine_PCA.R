library(readr)
wine <- read.csv(file.choose())

attach(wine)

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



###################### H_clust with PCA###############################

class(final)
wine_df <-  as.data.frame(final[,-1])

d <- dist(wine_df, method = "euclidean")

# Hclust method ,distance calc by euclideans
pca_hclust <- hclust(d,method="complete")
plot(pca_hclust,hang=-1)

# Converting the dendogram to have 3 clusters
wine_hclust1 <- cutree(pca_hclust, k = 3)
plot(wine_hclust1,hang=-1)

# created a matrix of wine_hclust1
matwine <- as.matrix(wine_hclust1)

# appended the matrix and original data read from csv as a dataframe
final_wine <- data.frame(matwine, wine)
View(final_wine)

# storing mean of the data for each cluster by aggregating by cluster
aggregate(wine, by = list(final_wine$matwine), FUN = mean)

library(readr)
write_csv(final, "hclustwinePCA.csv")