#haberman data
#https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival
#haberman's survival

dataset <- read.table("C:\\Users\\wspie\\OneDrive\\Documents\\R\\haberman.data", header=FALSE, sep =",")
head(dataset)
dim(dataset)
haberman <- dataset
head(haberman)
dim(haberman)

str(haberman)
names(haberman)

#rename columns
names(haberman)[1] <- "age"
names(haberman)[2] <- "year"
names(haberman)[3] <- "nodes"
names(haberman)[4] <- "survival_status"

names(haberman)

#scatterplot - age
plot(haberman[,"age"], haberman[,"positive_aux_nodes"], xlab = "age",
     ylab = "positive_aux_nodes", pch =20)

#histogram - age
hist(haberman[,"age"], main = "Histogram", xlab = "age",
     ylab = "count")

#boxplot - age
boxplot(haberman[,"age"], main="Boxplot", xlab= "age")

install.packages("corrplot")
library(corrplot)

B <- cor(haberman)
corrplot(B, method = "square")

head(haberman)

#linear regression
LM_model <- lm(age ~ survival_status, data = haberman)

print(LM_model)

LM_model_2 <- lm(survival_status ~ nodes, data = haberman)
print(LM_model_2)

LM_model_3 <- lm(survival_status ~ ., data = haberman)
print(LM_model_3)


