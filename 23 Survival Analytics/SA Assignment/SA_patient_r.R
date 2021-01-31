# Survivial Analytics in R
library(survminer)
library(survival)

pat <- read.csv(file.choose())

attach(pat)
str(pat)

class(pat)
unique(pat$Followup)
#  1.0  2.0  3.0  4.0  5.0  6.0  6.2  8.0  9.0 10.0 unique followup (time)

# Define variables 
time <- pat$Followup
event <- pat$Eventtype
group <- pat$Scenario  

# Descriptive statistics
summary(time)
table(event)
table(group)

# Kaplan-Meier non-parametric analysis ( Model Building part 1)
kmsurvival <- survfit(Surv(time, event) ~ 1,data =pat)

summary(kmsurvival)

plot(kmsurvival, xlab="Time", ylab="Survival Probability of Patient")

ggsurvplot(kmsurvival, data=pat, risk.table = TRUE)


# Kaplan-Meier non-parametric analysis by group ( Model building part 2)

kmsurvival1 <- survfit(Surv(time, event) ~ pat$Scenario, data= pat)
summary(kmsurvival1)

plot(kmsurvival1, xlab="Time", ylab="Survival Probability")

ggsurvplot(kmsurvival1, data=pat, risk.table = TRUE)

###############
surv_table <- data.frame(time = kmsurvival1$time,
                         n_risk = kmsurvival1$n.risk,
                         n_event = kmsurvival1$n.event,
                         n_censor = kmsurvival1$n.censor,
                         sur_prob = kmsurvival1$surv)

'
   time n_risk n_event n_censor  sur_prob
1   1.0     10       1        0 0.9000000
2   2.0      9       1        0 0.8000000
3   3.0      8       0        1 0.8000000
4   4.0      7       0        1 0.8000000
5   5.0      6       1        0 0.6666667
6   6.0      5       1        0 0.5333333
7   6.2      4       1        0 0.4000000
8   8.0      3       0        1 0.4000000
9   9.0      2       1        0 0.2000000
10 10.0      1       0        1 0.2000000

'
