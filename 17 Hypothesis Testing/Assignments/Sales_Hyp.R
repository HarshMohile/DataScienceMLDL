# Hypothesis Testing


# Check whether the diameter of both the donuts are same or not
# Ho --> No difference in Proportion
# H1 --> There is significant difference Proportion for Sales


library(readr)
library(dplyr)
library(data.table)

sales <-  read.csv(file.choose())

#t_sales <- transpose(sales)
#colnames(t_sales) <- rownames(sales)
#rownames(t_sales) <- colnames(sales)

class(sales)

row.has.na <- apply(lab, 1, function(x){any(is.na(x))})
sum(row.has.na)


########### Proportional Z Test ##########
attach(sales)
names(sales)

#install.packages("BSDA")
library(BSDA)



sd(sales$East)
# 272.2361

sd(sales$West)
#976.5145

sd(sales$North)
#866.2058

sd(sales$South)
#480.8326


t.test(sales$East, sales$West)



#t = -0.82307, df = 1.1545, p-value = 0.5461

#0.5461 >0.05 Null Hypothesis Accepted

#  Conclusion::No difference in  Proportion of sales in any EAST and WEST Direction by Men and Women

t.test(sales$North, sales$South)

#t = -0.82307, df = 1.1545, p-value = 0.6921

#0.6921 >0.05 Null Hypothesis Accepted

#  Conclusion::No difference in  Proportion of sales in any north and south Direction by Men and Women

##### Using ChiSquare test #############

attach(sales)


chisq.test(sales$East,sales$West,correct=FALSE)
#X-squared = 2, df = 1, p-value = 0.1573 Accept Null hypothesis

chisq.test(sales$North,sales$South,correct=FALSE)
#X-squared = 2, df = 1, p-value = 0.1573 Accept Null hypothesis


## Variance Test 
var.test(East, West) #0.07772023 
#F = 0.07772, num df = 1, denom df = 1, p-value = 0.3462..Accept null hypothesis 


