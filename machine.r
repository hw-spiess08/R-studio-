#Computer hardward data set from UCI
#Using Caret
#https://archive.ics.uci.edu/ml/machine-learning-databases/machine/

library(caret)
library(ranger)
library(tidyverse)
library(e1071)
library(data.table)

# load in machine dataset\

machine <- read.table("C:/Users/wspie/OneDrive/Documents/R/Data_sets/machine.data",
                      sep = ",")

head(machine)

# load in column names
colnames(machine) <- c("Vendor", "Model", "MYCT", "MMIN", 
                       "MMAX", "CACH", 
                       "CHMIN", "CHMAX", "PRP", "ERP")

head(machine)

dim(machine)
str(machine)


library(skimr)
#summary statistics
skim(machine)

#check what categories are factors
sapply(machine, class)

sapply(machine, is.factor)

machine_matrix <- cor(machine[sapply(machine, function(x) !is.factor(x))])

library(corrplot)
#correlation plot
corrplot(cor(machine_matrix), method = "square")

library("PerformanceAnalytics")
#different correlation charts
chart.Correlation(machine_matrix)

# split into training and testing
set.seed(23489)
train_index <- sample(1:nrow(machine), 0.9 * nrow(machine))
machine_train <- machine[train_index, ]
machine_test <- machine[-train_index, ]

# remove the original dataset
rm(machine)

# view the first 6 rows of the training data
head(machine_train)

#size of training set
dim(machine_train)

# fit a random forest model (using ranger)
rf_fit <- train(as.factor(old) ~ ., 
                data = machine_train, 
                method = "ranger")

rf_fit
library(ggplot2)
ggplot(varImp(rf_fit))

#MYCT = Machine cycle time
# fit a random forest model (using ranger)
rf_fit <- train(MYCT ~ ., 
                data = machine_train, 
                method = "ranger")

rf_fit

# predict the outcome on a test set
machine_rf_pred <- predict(rf_fit, machine_test)

# compare predicted outcome and true outcome
confusionMatrix(machine_rf_pred, as.factor(machine_test$MYCT))
#confusion matrix not working...

#Yeo Johnson transformation

machine_no_nzv_pca <- preProcess(select(machine_train, -MYCT),
                                 method = c("center", "scale", "YeoJohnson", "nzv", "pca"))

machine_no_nzv_pca

#identify which variables were ignored, centered, scaled, etc...
machine_no_nzv_pca$method

#identify principal components
machine_no_nzv_pca$rotation

#Data Splitting
#Data partion

#kfold 
machine_grouped <- cbind(machine_train[1:50, ], group = 
                           rep(1:10, each =5))

head(machine_grouped, 10)
#perform grouped K means
group_folds <- groupKFold(machine_grouped$group, k=10)
group_folds

#partition
set.seed(998)
# create a testing and training set
in_training <- createDataPartition(machine_train$MYCT, p = .75, list = FALSE)
training <- machine_train[ in_training,]
testing  <- machine_train[-in_training,]

fit_control <- trainControl(method = "cv",
                            number = 10)

set.seed(825)
rf_fit <- train(as.factor(MYCT) ~ ., 
                data = machine_train, 
                method = "ranger",
                trControl = fit_control)
rf_fit

# specify that the resampling method is 
group_fit_control <- trainControl(## use grouped CV folds
  index = group_folds,
  method = "cv")

#breaks down here...
set.seed(825)
rf_fit <- train(as.factor(MYCT) ~ ., 
                data = select(machine_grouped, - group), 
                method = "ranger",
                trControl = group_fit_control)

rf_fit

# define a grid of parameter options to try
rf_grid <- expand.grid(mtry = c(2, 3, 4, 5),
                       splitrule = c("gini", "extratrees"),
                       min.node.size = c(1, 3, 5))
rf_grid

# re-fit the model with the parameter grid
rf_fit <- train(as.factor(MYCT) ~ ., 
                data = select(machine_grouped, - group), 
                method = "ranger",
                trControl = group_fit_control,
                # provide a grid of parameters
                tuneGrid = rf_grid)
rf_fit









