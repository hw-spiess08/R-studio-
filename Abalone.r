#Abalone - from UCI data
#following CARET tutorial
#http://www.rebeccabarter.com/blog/2017-11-17-caret_tutorial/


library(caret)
library(ranger)
library(tidyverse)
library(e1071)
library(data.table)
# load in abalone dataset\

abalone <- read.table("C:/Users/wspie/OneDrive/Documents/R/abalone.data",
                    sep = ",")


# load in column names
colnames(abalone) <- c("sex", "length", "diameter", "height", 
                            "whole.weight", "shucked.weight", 
                            "viscera.weight", "shell.weight", "age")
                            
head(abalone)
dim(abalone)
str(abalone)


library(skimr)
#summary statistics
skim(abalone)

#correlation plot
#corrplot(cor(abalone), method = "square")

#different correlation charts
#chart.Correlation(abalone)


# add a logical variable for "old" (age > 10)
abalone <- abalone %>%
  mutate(old = age > 10) %>%
  # remove the "age" variable
  select(-age)

# split into training and testing
set.seed(23489)
train_index <- sample(1:nrow(abalone), 0.9 * nrow(abalone))
abalone_train <- abalone[train_index, ]
abalone_test <- abalone[-train_index, ]
# remove the original dataset
rm(abalone)
# view the first 6 rows of the training data
head(abalone_train)

#size of training set
dim(abalone_train)

# fit a random forest model (using ranger)
rf_fit <- train(as.factor(old) ~ ., 
                data = abalone_train, 
                method = "ranger")

rf_fit

# predict the outcome on a test set
abalone_rf_pred <- predict(rf_fit, abalone_test)
# compare predicted outcome and true outcome
confusionMatrix(abalone_rf_pred, as.factor(abalone_test$old))

#confusion matrix not working...

#Yeo Johnson transformation

abalone_no_nzv_pca <- preProcess(select(abalone_train, -old),
                        method = c("center", "scale", "YeoJohnson", "nzv", "pca"))

abalone_no_nzv_pca

#identify which variables were ignored, centered, scaled, etc...
abalone_no_nzv_pca$method

#identify principal components
abalone_no_nzv_pca$rotation

#Data Splitting
#Data partion


#kfold 
abalone_grouped <- cbind(abalone_train[1:50, ], group = 
                   rep(1:10, each =5))

head(abalone_grouped, 10)
#perform grouped K means
group_folds <- groupKFold(abalone_grouped$group, k=10)
group_folds

#Resampling - traincontrol

set.seed(998)
# create a testing and training set
in_training <- createDataPartition(abalone_train$old, p = .75, list = FALSE)
training <- abalone_train[ in_training,]
testing  <- abalone_train[-in_training,]

fit_control <- trainControl(method = "cv",
                            number = 10)

#random forest model
set.seed(825)
rf_fit <- train(as.factor(old) ~ ., 
                data = abalone_train, 
                method = "ranger",
                trControl = fit_control)
rf_fit

# specify that the resampling method is 
group_fit_control <- trainControl(## use grouped CV folds
  index = group_folds,
  method = "cv")

#breaks down here...
set.seed(825)
rf_fit <- train(as.factor(old) ~ ., 
                data = select(abalone_grouped, - group), 
                method = "ranger",
                trControl = group_fit_control)

rf_fit

# define a grid of parameter options to try
rf_grid <- expand.grid(mtry = c(2, 3, 4, 5),
                       splitrule = c("gini", "extratrees"),
                       min.node.size = c(1, 3, 5))
rf_grid

# re-fit the model with the parameter grid
rf_fit <- train(as.factor(old) ~ ., 
                data = select(abalone_grouped, - group), 
                method = "ranger",
                trControl = group_fit_control,
                # provide a grid of parameters
                tuneGrid = rf_grid)
rf_fit












































