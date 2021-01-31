library(readr)
wine <- read.csv(file.choose())

normalized_data <- scale(wine)
View(normalized_data)


d <- dist(normalized_data, method = "euclidean")

# Hclust method ,distance calc by euclideans
wine_hclust <- hclust(d,method="complete")
plot(wine_hclust,hang=-1)

# Converting the dendogram to have 3 clusters
wine_hclust1 <- cutree(wine_hclust, k = 3)
plot(wine_hclust1,hang=-1)

# created a matrix of wine_hclust1
matwine <- as.matrix(wine_hclust1)

# appended the matrix and original data read from csv as a dataframe
final <- data.frame(matwine, wine)
View(final)

# storing mean of the data for each cluster by aggregating by cluster
aggregate(wine, by = list(final$matwine), FUN = mean)

library(readr)
write_csv(final, "hclustwine.csv")


#Group.1     Type  Alcohol    Malic      Ash Alcalinity Magnesium  Phenols Flavanoids
#1       1 1.516393 12.96770 1.944344 2.368607   18.88115 100.65574 2.563115  2.5490164
#2       2 2.000000 12.56200 1.410000 1.722000   15.62000  92.40000 1.974000  1.6260000
#3       3 2.941176 13.12235 3.364902 2.424706   21.34314  98.27451 1.685490  0.8254902
#Nonflavanoids Proanthocyanins    Color       Hue Dilution  Proline
#1     0.3281148        1.793852 4.238279 1.0602951 2.995984 807.9098
#2     0.2880000        1.032000 3.490000 1.1940000 2.372000 536.0000
#3     0.4498039        1.160196 7.172941 0.6882353 1.715882 621.6078
