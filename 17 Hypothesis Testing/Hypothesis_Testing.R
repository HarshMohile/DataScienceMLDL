# Hypothesis Testing

# Load the Dataset
library(readxl)

# 2 sample t Test

######## Promotion.xlsx data ##########
# Business Problem: Which offer is better to launch?
# Ho: Purchases made using FWI </= Purchases made using SC  --> no action
# Ha: Purchases made using FWI > Purchases made using SC    --> take action

Promotion <- read_excel(file.choose())
View(Promotion)
colnames(Promotion) <- c("InterestRateWaiver", "StandardPromotion")

# Changing column names
View(Promotion)
attach(Promotion)

# Normality test
# Ho: Data are normal
# Ha: Data are not normal --> take action = transformation

shapiro.test(InterestRateWaiver)
# p-value = 0.2246 > 0.05 so p high null fly => It follows normal distribution

shapiro.test(StandardPromotion)
# p-value = 0.1916 > 0.05 so p high null fly => It follows normal distribution

# Variance test
var.test(InterestRateWaiver, StandardPromotion)
# p-value = 0.653 > 0.05 so p high null fly => Equal variances


# 2 sample t Test assuming equal variances
t.test(InterestRateWaiver, StandardPromotion, alternative = "two.sided", conf.level = 0.95, var.equal = T)

# alternative = "two.sided" means we are checking for equal and unequal means
# null Hypothesis -> Equal means
# Alternate Hypothesis -> Unequal Hypothesis
# p-value = 0.02523 < 0.05 accept alternate Hypothesis unequal means

?t.test
t.test(InterestRateWaiver, StandardPromotion, alternative = "greater")

# alternative = "greater means true difference is greater than 0
# Null Hypothesis -> (InterestRateWaiver-StandardPromotion) < 0
# Alternative Hypothesis -> (StandardPromotion - InterestRateWaiver) > 0
# p-value = 0.01211 < 0.05 => p low null go => accept alternate hypothesis

# Conclusion:
# InterestRateWaiver better promotion than StandardPromotion


########## Anova ##########

library(readxl)
# Load the data: Contract_Renewal Data
CRD <- read_excel(file.choose())
View(CRD)
attach(CRD)


# Normality test
shapiro.test(`Supplier A`)
shapiro.test(`Supplier B`)
shapiro.test(`Supplier C`)

# Variance test
var.test(`Supplier A`, `Supplier B`)
var.test(`Supplier B`, `Supplier C`)
var.test(`Supplier C`, `Supplier A`)

Stacked_Data <- stack(CRD)
?stack
View(Stacked_Data)

attach(Stacked_Data)
colnames(Stacked_Data)

Anova_results <- aov(values ~ ind, data = Stacked_Data)
summary(Anova_results)

# p-value = 0.104 > 0.05 accept null hypothesis
# 3 suppliers transaction times are equal


########### Proportional T Test ##########
library(readxl)

# Load the data: JohnyTalkers data
Johnytalkers <- read_excel(file.choose()) 
View(Johnytalkers)

attach(Johnytalkers)

table1 <- table(Person)
table1
table2 <- table(Drinks)
table2
table3 <- table(Person, Drinks)
table3

?prop.test
prop.test(x = c(58,152), n = c(480,740), conf.level = 0.95, alternative = "two.sided")
# two.sided -> means checking for equal proportions of Adults and children under purchased
# p-value = 6.261e-05 < 0.05 accept alternate hypothesis i.e.
# Unequal proportions 

prop.test(x = c(58,152), n = c(480,740), conf.level = 0.95, alternative = "greater")
# Ha -> Proportions of Adults > Proportions of Children
# Ho -> Proportions of Children > Proportions of Adults
# p-value = 0.999 > 0.05 accept null hypothesis 
# so proportion of Children > proportion of children 
# Do not launch the incentive program


######### Chi Square Test ##########
library(readxl)

# Load the data: Bahaman.xlsx
Bahaman <- read_excel(file.choose()) 
View(Bahaman)

attach(Bahaman)
table(Country, Defective)

chisq.test(table(Country, Defective))

# p-value = 0.6315 > 0.05  => Accept null hypothesis
# => All countries have equal proportions 

# All Proportions are equal 
