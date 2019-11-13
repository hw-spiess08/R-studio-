#Machine learning & caret
#https://www.machinelearningplus.com/machine-learning/caret-package/


install.packages(c('caret', 'skimr', 'RANN', 'randomForest', 'fastAdaboost', 'gbm', 'xgboost', 'caretEnsemble', 'C50', 'earth')

install.packages("skimr")                 
                                  
library(caret)
library(skimr)
                 
# Import dataset
orange <- read.csv('https://raw.githubusercontent.com/selva86/datasets/master/orange_juice_withmissing.csv')

str(orange)

head(orange)

#split data for test & training - Use dataPartition

#random number generator
set.seed(100)

#training data - get row #'s for training
trainRowNumbers <- createDataPartition(orange$Purchase, p =0.8, list = FALSE)

#define 'train' dataset
trainData <- orange[trainRowNumbers,]

#define 'test' dataset
testData <- orange[-trainRowNumbers,]

x = trainData[, 2:18]

y = trainData$Purchase

skimmed <- skim_to_wide(trainData)
skimmed[, c(1:5, 9:11, 13, 15:16)]

#missing values in the data set, will predict missing values using the
#'k-Nearest Neighbors'
#'Part of pre-processing
# will be set as 'method = knnImpute'
#then use 'predict()' on the created preprocess

#Create the knn imputation model on the training data
library(caret)
preProcess_missing_data_model <- preProcess(trainData, method = 'knnImpute')
preProcess_missing_data_model

#use model to predict missing values in TrainData

library(RANN)

#RANN required for knnImpute

trainData <- predict(preProcess_missing_data_model, newdata = trainData)
anyNA(trainData)

#One-Hot encoding (dummy variables)
#use for categorical variables, example colors
#create categorical variable to as many binary variables as there are categories
#use 'dummyVars()'

dummies_model <- dummyVars(Purchase~., data= trainData)

# Create the dummy variables using predict. The Y variable (Purchase) will not be present in trainData_mat.
trainData_mat <- predict(dummies_model, newdata = trainData)

#convert to dataframe
trainData <- data.frame(trainData_mat)

str(trainData)

#preprocess
#convert the numeric variables to range between 1 and 0
#method = 'range' normalizes values to between 1 and 0

preProcess_range_model <- preProcess(trainData, method = "range")
trainData <- predict(preProcess_range_model, newdata = trainData)

#append the Y variable
trainData$Purchase <- y

apply(trainData[, 1:10], 2, FUN = function(x){c('min'=min(x), 'max'=max(x))})

#'featurePlot' a easy way to see how predictors influence the Y
#use box plot

featurePlot(x = trainData[, 1:18], 
            y = trainData$Purchase,
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x=list(relation="free"),
                          y=list(relation="free")))

#density plot
#note density on 'weekofpurchase' and 'LoyalCH'

featurePlot(x = trainData[, 1:18], 
            y = trainData$Purchase, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

#recursive feature elimination (RFE)
# 3 steps
# build a model on training dataset & estimate feature importances on test dataset
# iterate by building models of the most important features and subset sizes, ranking predictors..
#model performances are compared across different subset size to find the optimal number & list of final predictors

set.seed(100)
options(warn = -1)

subsets <- c(1:5, 10, 15, 18)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

lmProfile <- rfe(x=trainData[, 1:18], y = trainData$Purchase,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile

#see available algorithms in caret
modelnames <- paste(names(getModelInfo()), collapse = ', ')
modelnames

#use model method 'earth'

modelLookup('earth')

#set the seed for reproductibility
set.seed(1000)

#train model using randomForest and predict on the training data itself
model_mars = train(Purchase~., data=trainData, method='earth')
fitted <- predict(model_mars)

model_mars

plot(model_mars, main = "Model Accuracies with MARS")

#extract variable importance using 'varImp()'

varimp_mars <- varImp(model_mars)
plot(varimp_mars, main = "Variable Importance with MARS")

#good to here---------

#preProcess test data
#preprocess
#convert the numeric variables to range between 1 and 0
#method = 'range' normalizes values to between 1 and 0
#-------------------------------------------------

preProcess_range_model <- preProcess(testData, method = "range")
testData <- predict(preProcess_range_model, newdata = testData)

#append the Y variable
testData$Purchase <- y

apply(testData[, 1:10], 2, FUN = function(x){c('min'=min(x), 'max'=max(x))})

#'featurePlot' a easy way to see how predictors influence the Y
#use box plot

featurePlot(x = testData[, 1:18], 
            y = testData$Purchase,
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x=list(relation="free"),
                          y=list(relation="free")))
#----------------------------------------------------
#..not working exactly as expected...

#prepare the test dataset and predict
# 1 impute missing values
testData2 <- predict(preProcess_missing_data_model, testData)

# 2 create one-hot encodings (dummy variables)
testData3 <- predict(dummies_model, testData2)

# 3 Transform the features to range between 0 and 1
testData4 <- predict(preProcess_range_model, testData3)

#view
head(testData4[, 1:10])

#predict on testData
predicted <- predict(model_mars, testData4)
head(predicted)

#compute the confusion matrix
confusionMatrix(reference = testData$Purchase, data = predicted, mode = 
                  'everything', positive = 'MM')

#Tuning to optimize the model using the 'train()'
#can set the tunelength
#define & set the tuneGrid

#define the training control
fitControl <- trainControl (
  method = 'cv', 
  number = 5,
  savePredictions = 'final',
  classProbs = T,
  summaryFunction = twoClassSummary
)

#Hyper Parameter Tuning using 'tuneLength'

set.seed(100)

model_mars2 = train(Purchase~., data = trainData, method='earth', 
                    tuneLength = 5, metric='ROC',
                    trControl=fitControl)
model_mars2

#Can also tune using 'tuneGrid'instead of 'tuneLength'

#step 1: Define the tuneGrid
#marsGrid <- expand.grid(nprune = c(2,4,6,8,10),
#                        degree = c(1,2,3))

#step 2: Tune hyper parameter by setting 'tuneGrid'
#...not working....
#set.seed(100)
#model_mars3 = train(Purchase~., data = trainData, method='earth', 
#                    metric='ROC', tuneGrid = marsGrid, trConrol = fitControl)
#model_mars3

#Evaluate performance from multiple models

#training Adaboost

set.seed(100)
#train model using adaboost
model_adaboost = train(Purchase~., data = trainData, method = 'adaboost', 
                       tuneLenght=2, trControl = fitControl)

model_adaboost

#training random forest

set.seed(100)

model_rf = train(Purchase~., data = trainData, method = 'rf', tuneLength = 5,
                 trControl = fitControl)

model_rf

#train using MARS - xgboostDart....takes awhile to run

install.package('xgboost')
library(xgboost)

set.seed(100)

model_xgbDART = train(Purchase~., data = trainData, method = 'xgbDART',
                      tuneLength=5, trControl = fitControl, 
                      verbose=F)

model_xgbDART


#training SVM...takes awhile to run
set.seed(100)

# Train the model using MARS
model_svmRadial = train(Purchase ~ ., data=trainData, method='svmRadial', tuneLength=15, trControl = fitControl)
model_svmRadial


#Run 'resamples' to compare the models

# Compare model performances using resample()
models_compare <- resamples(list(ADABOOST=model_adaboost, RF=model_rf, XGBDART=model_xgbDART, MARS=model_mars2, SVM=model_svmRadial))

# Summary of the models performances
summary(models_compare)

# Draw box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)


#ensemble predictions from multiple models

install.packages("caretEnsemble")
library(caretEnsemble)

# Stacking Algorithms - Run multiple algos in one call.
trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             classProbs=TRUE)

algorithmList <- c('rf', 'adaboost', 'earth', 'xgbDART', 'svmRadial')

set.seed(100)

models <- caretList(Purchase ~ ., data=trainData, trControl=trainControl, methodList=algorithmList) 

#takes awhile....
results <- resamples(models)
summary(results)

#plot results
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

#combine predictions of multiple models to form a final prediction
#use 'caretStack'

#Create the trainContrl
set.seed(101)

stackControl <- trainControl(method = "repeatedcv",
                             number=10,
                             repeats=3,
                             savePredictions=TRUE,
                             classProbs=TRUE)

#ensemble the predictions of 'models' to form a new combined prediction based on glm

stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)

#predict on testData
stack_predicteds <- predict(stack.glm, newdata = testData4)
head(stack_predicteds)











