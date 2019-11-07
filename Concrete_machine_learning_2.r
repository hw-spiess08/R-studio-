
# Import dataset
#use file, import, update

install.packages("tidyverse")

library(tidyverse)
library(readxl)

Concrete_Data <- read_excel("~/R/Concrete_Data.xls")


head(Concrete_Data)

concrete <- (Concrete_Data)

head(concrete)

str(concrete)

#size of dataset
dim(concrete)

names(concrete)

#strength is the variable trying to predict, y variable

#load libraries

library(dplyr)
library(ggplot2)
library(PerformanceAnalytics)

install.packages('ggthemes')

library(ggthemes)
library(corrplot)

install.packages('car')

library(car)
library(psych)
library(caret)
library(caretEnsemble)

install.packages("doParallel")
library(doParallel)

class(concrete)

names(concrete)

#Exploratory data analysis
#view displays the data in a table form
View(concrete)

glimpse(concrete)

head(concrete)


#use different views to make sure each row is an individual observation & has data

#need to rename columns to simplify

concrete = rename(concrete, cement = `Cement (component 1)(kg in a m^3 mixture)`)
summary(concrete$cement)

concrete = rename(concrete, slag = `Blast Furnace Slag (component 2)(kg in a m^3 mixture)`)
summary(concrete$slag)

concrete = rename(concrete, ash = `Fly Ash (component 3)(kg in a m^3 mixture)`)
summary(concrete$ash)

concrete = rename(concrete, water = `Water  (component 4)(kg in a m^3 mixture)`)
summary(concrete$water)

concrete = rename(concrete, superplastic = `Superplasticizer (component 5)(kg in a m^3 mixture)`)
summary(concrete$superplastic)

concrete = rename(concrete, coarseagg = `Coarse Aggregate  (component 6)(kg in a m^3 mixture)`)
summary(concrete$coarseagg)

concrete = rename(concrete, fineagg = `Fine Aggregate (component 7)(kg in a m^3 mixture)`)
summary(concrete$fineagg)

concrete = rename(concrete, age = `Age (day)`)
summary(concrete$age)

concrete = rename(concrete, strength = `Concrete compressive strength(MPa, megapascals)`)
summary(concrete$strength)

glimpse(concrete)
View(concrete)

summary(concrete$strength)


getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]}

ggplot(data = concrete) +
  geom_histogram(mapping = aes(x = strength), bins = 15, boundary = 0, fill = "gray", col = "black") +
  geom_vline(xintercept = mean(concrete$strength), col = "blue", size = 1) +
  geom_vline(xintercept = median(concrete$strength), col = "red", size = 1) +
  geom_vline(xintercept = getmode(concrete$strength), col = "green", size = 1) +
  annotate("text", label = "Median = 34.4", x = 23, y = 100, col = "red", size = 5) +
  annotate("text", label = "Mode = 33.4", x = 23, y = 125, col = "black", size = 5) +
  annotate("text", label = "Mean = 35.8", x = 45, y = 45, col = "blue", size = 5) +
  ggtitle("Histogram of strength") + 
  theme_bw()

summary(concrete)


#correlation plot - correlation between all variables 
corrplot(cor(concrete), method = "square")

#another correlation plot
chart.Correlation(concrete)


#check for NAs
anyNA(concrete)

#another way to check for NAs
sapply(concrete, {function(x) any(is.na(x))})

#look for outliers using boxplot
boxplot(concrete[-9], col = "orange", main = "Features Boxplot")

#focus on age to look closely at outliers
boxplot(concrete$age, col = "red")

#drill into '4' outliers
age_outliers <- which(concrete$age > 100)
concrete[age_outliers, "age"]
#---62 outliers total


#'variance inflation factor' = 'vif()'
#VIF score > 10 indicates strong likelyhood of multicollinearity
#VIF score > 10 may indicate variable needs to go
#simple linear model

simple_lm <- lm(strength~., data = concrete)
vif(simple_lm)

#drop features that are not adding value
par(mfrow = c(2,2))

hist(concrete$age)
hist(concrete$superplastic)
hist(log(concrete$age), col = "red")
hist(log(concrete$superplastic), col = "red")

#converting age to log value
#converting superplastic to log value
#superplastic values that are 0, trouble taking log, so manually set to 0

concrete$age <- log(concrete$age)
concrete$superplastic <- log(concrete$superplastic)
concrete$superplastic <- ifelse(concrete$superplastic == -Inf, 0,  
                                concrete$superplastic)
head(concrete)

#drop features that are not adding value

concrete$ash <- NULL
head(concrete)

#K-fold cross-validation, use 'train' function

fitcontrol <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10)

model.cv <- train(strength~.,
                  data = concrete,
                  method = "lasso",
                  trcontrol = fitcontrol)

model.cv

#preprocess - center data & scale data

model.cv <- train(strength~., 
                  data = concrete,
                  method = "lasso", 
                  trControl = fitcontrol, 
                  preProcess = c('scale', 'center'))

model.cv

#hyper parameters using TuneGrid
#use 'expand.grid'

lambdaGrid <- expand.grid(lambda = 10^seq(10, -2, length=100))

model.cv <- train(strength ~.,
                  data = concrete,
                  method = "ridge",
                  trControl = fitcontrol, 
                  preProcess = c('scale', 'center'),
                  tuneGrid = lambdaGrid, 
                  na.action = na.omit)

model.cv

#Can also use search = "random" to test a range of values
fitcontrol <- trainControl(method = "repeatedcv", 
                           number = 10,
                           repeats = 10, 
                           search = "random")

model.cv

#check variable importance
#ggplot(varImp(model.cv))

#predictions
predictions <- predict(model.cv, concrete)

predictions














