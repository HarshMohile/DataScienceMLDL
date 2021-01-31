# Hypothesis Testing


# Check whether the diameter of both the donuts are same or not
# Ho --> No difference in diameter
# H1 --> There is significant difference in diameter of 2 donuts

# Usng Z test as number of samples are more than 30 
library(readr)
library(dplyr)

ct <-  read.csv(file.choose())

class(ct)

row.has.na <- apply(ct, 1, function(x){any(is.na(x))})
sum(row.has.na)

ct <- na.omit(ct) 

########### Proportional Z Test ##########
attach(ct)
names(ct)
install.packages("BSDA")
library(BSDA)



sd(ct$Unit.A)
# 0.2884085

sd(ct$Unit.B)
#0.3434006

??z.test
z.test(ct$Unit.A, ct$Unit.B, alternative = "two.sided", mu = 0, sigma.x =  0.2884085,
       sigma.y = 0.3434006, conf.level = 0.95)

#data:  ct$Unit.A and ct$Unit.B
#z = 0.72287, p-value = 0.4698
#0.4698 >0.05 Null Hypothesis Accepted

#  Conclusion::No difference in diameter of donuts

##### Using ChiSquare test#############
table1 <- table(Unit.A)
table1
table2 <- table(Unit.B)
table2
table3 <- table(Unit.A, Unit.B)
table3

chisq.test(table3)
#data:  table3
#X-squared = 1190, df = 1156, p-value = 0.2376

#0.2376 >0.05 so null hypothesis accepted

### Normality test 
attach(ct)
# Normality test
shapiro.test(ct$Unit.A)
#W = 0.96495, p-value = 0.32  .Accept null hypothesis 


## Variance Test 
var.test(ct$Unit.A, ct$Unit.B) #0.7053649
#p-value = 0.3136  null hypothesis accepted


