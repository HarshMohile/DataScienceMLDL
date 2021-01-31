# Loading Universities data

library(readxl)
input <- read_excel(file.choose())
mydata <- input[ ,c(1, 3:8)]
# mydata <- input[ , -2]

## the first column in mydata has university names
data <- mydata[, -1]
attach(data)

?princomp
pcaObj <- princomp(data, cor = TRUE, scores = TRUE, covmat = NULL)

str(pcaObj)
summary(pcaObj)

loadings(pcaObj)

plot(pcaObj) # graph showing importance of principal components 

biplot(pcaObj)

plot(cumsum(pcaObj$sdev * pcaObj$sdev) * 100 / (sum(pcaObj$sdev * pcaObj$sdev)), type = "b")

pcaObj$scores
pcaObj$scores[, 1:3]

# Top 3 pca scores 
final <- cbind(input[, 1], pcaObj$scores[, 1:3])
View(final)

# Scatter diagram
plot(final$Comp.1, final$Comp.2)
