library(readxl)
flight <- read_excel(file.choose())

myflight <- flight[ , c(-1)]
View(myflight)

normalized_data <- scale(myflight)
View(normalized_data)

d <- dist(normalized_data, method = "euclidean")

flight_hclust <- hclust(d,method="complete")
plot(flight_hclust,hang=-1)

flight_hclust <- cutree(flight_hclust, k = 3)


matflight <- as.matrix(flight_hclust)

final <- data.frame(matflight, myflight)
View(final)


aggregate(myflight, by = list(final$matflight), FUN = mean)

library(readr)
write_csv(final, "hclustflight.csv")


