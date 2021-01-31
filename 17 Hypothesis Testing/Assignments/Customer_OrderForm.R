# Hypothesis Testing


# Check whether the 
# Ho --> No error /defect in Customer form
# H1 --> There is significant  error in Customer Form 


library(readr)
library(dplyr)

cust <-  read.csv(file.choose())

class(cust)

row.has.na <- apply(cust, 1, function(x){any(is.na(x))})
sum(row.has.na)

## countries are in their own columns; so we need to stack the data
stacked_cof<-stack(cust)

View(stacked_cof)


### Using Chi Square test ####
# Chi Square because it has categorical ,since chisq works with categorical and numerical

chisq.test(table(stacked_cof$ind,stacked_cof$values))

#data:  table(stacked_cof$ind, stacked_cof$values)
#X-squared = 3.859, df = 6, p-value = 0.6958 So
## Result :Since 0.69 > 0.005 (alpha) we accept null hypothesis which there is no error in customer form or defect







