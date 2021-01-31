# Hypothesis Testing


# Check whether the TAT of labs matter if they are different avg TAT
# Ho --> No difference in TurnAround Time for labs
# H1 --> There is significant difference TAT for labs


library(readr)
library(dplyr)

lab <-  read.csv(file.choose())

class(lab)

row.has.na <- apply(lab, 1, function(x){any(is.na(x))})
sum(row.has.na)


########### Proportional Z Test ##########
attach(lab)
names(lab)

#install.packages("BSDA")
library(BSDA)



sd(lab$Laboratory_1)
# 13.91967

sd(lab$Laboratory_2)
#14.95711

sd(lab$Laboratory_3)
#15.7948

sd(lab$Laboratory_4)
#15.08508

??z.test
z.test(lab$Laboratory_1, lab$Laboratory_2, alternative = "two.sided", mu = 0, 
       sigma.x =  13.91967,
       sigma.y = 14.95711, conf.level = 0.95)

#z = -0.34612, p-value = 0.7293

#0.7293 >0.05 Null Hypothesis Accepted

#  Conclusion::No difference in  TAT for labs

##### Using ChiSquare test#############

table1 <- table(Laboratory_1)
table1
table2 <- table(Laboratory_2)
table2
table3 <- table(Laboratory_1, Laboratory_2)
table3

chisq.test(table3)
#data:  table3
#X-squared = 13720, df = 13570, p-value = 0.1811

#0.1811 >0.05 so null hypothesis accepted


######## Normality test 
attach(lab)
# Normality test
shapiro.test(Laboratory_1)
#W = 0.98867, p-value = 0.4232  .Accept null hypothesis 


## Variance Test 
var.test(Laboratory_1, Laboratory_2) #0.8660883 
#F = 0.86609, num df = 119, denom df = 119, p-value = 0.4341..Accept null hypothesis 


