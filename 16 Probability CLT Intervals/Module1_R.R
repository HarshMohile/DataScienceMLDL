library(readr)

cars <- read.csv(file.choose())

mpg <- cars$MPG

pnorm(mpg>38)
