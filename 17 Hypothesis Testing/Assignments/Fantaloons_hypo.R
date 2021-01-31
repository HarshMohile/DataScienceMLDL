#Hypothesis testing 

# Check whether the 
# Ho --> No change in stores in which males go into  (by day)
# H1 --> There is significant  change in % for males v females walking in different stores

fan <- read.csv(file.choose())
class(fan)

fan$Weekdays <- as.factor(fan$Weekdays)
fan$Weekend <- as.factor(fan$Weekend)

View(fan)
summary(fan)
# looking at summary we get 
'Weekdays     Weekend   
: 25         : 25  
Female:287   Female:233  
Male  :113   Male  :167  '

fanta2 <- data.frame("Weekdays"=c(287,113), "Weekend" = c(233,167))

row.names(fanta2) <- c("Female","Male")

chisq.test(fanta2)
#data:  fanta2
#X-squared = 15.434, df = 1, p-value = 8.543e-05  p value less than 0.05 .So we reject the null hypothesis 
# There is a shift in the  Malesv females going to different stores by % with 95% confidence
