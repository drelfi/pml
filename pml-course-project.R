#' This script is a potential solution to the Practical Machine Learning course
#' Available through Coursera from the Johns Hopkins University
#' 
#' Six young health participants were asked to perform one set of 10 repetitions 
#' of the Unilateral Dumbbell Biceps Curl in five different fashions: 
#'   exactly according to the specification (Class A), 
#'   throwing the elbows to the front (Class B), 
#'   lifting the dumbbell only halfway (Class C), 
#'   lowering the dumbbell only halfway (Class D) 
#'   and throwing the hips to the front (Class E).
#'
#' Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3gV7OiQkO

get.pml.files <- function() {
  
  if (! file.exists("pml-training.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv", method="curl")
  }
  if (!file.exists("pml-testing.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml-testing.csv", method="curl")
  }
  
  if (! exists('training')) {
    training <- read.csv('pml-training.csv', 
                         header=TRUE, 
                         na.strings=c("","#DIV/0!","NA")
                         #stringsAsFactors=FALSE
                         )
  }
    
  training
}

remove_na_columns <- function(training) {
  require('dplyr')
  
  # Called my attention that many (67) columns has exactly 19216 NAs
  nas <- apply(training, 2, function(x) {sum(is.na(x))})
  colnas <- data.frame(nas)
  colnas['colname'] <- row.names(colnas)
  colnames_nas <- colnas %>% filter(nas > 0) %>% select(colname)
  
  # Get the training without NAs cols
  if (dim(colnames_nas)[1] == 0) {
    training
  }
  else {
    training[, -which(names(training) %in% colnames_nas[['colname']])]
  }
  
  # There are 406 observations with no NAs. 
  # We should research if the obs without NAs can create
  # a better model than doing it with all obs
}

# Run on the 406 observations with no NAs
# Try to filter data to get the most of the variance (data reduction)
# How to save the model once it is done

set.seed(48293)
training <- remove_na_columns(get.pml.files())

require('caret')

# Create a set of training and testing set to measure error
if (!exists('pmlTraining')) {
  inTrain <- createDataPartition(y=training$classe,p=0.75,list=FALSE)

  pmlTest <- training[-inTrain,]    
  pmlTraining <- training[inTrain,]
}

# Clean the columns to only those needed
if ( dim(pmlTraining)[2] >= 60 ) {
  pmlTraining <- pmlTraining[,-which( names(pmlTraining) %in% 
                                        c('user_name', 'cvtd_timestamp', 'new_window', 'X'))]

}

# Copied from project instructions to store the results to submit
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


# Since I wanted to try the caret train function with boosting, I decided to do it in parallel
library(doMC)
registerDoMC(cores = 3)

# although the train method does it for you, let's see how many variables could cover the 90% of variance in the data
preProc <- preProcess(
  pmlTraining[,-which( names(pmlTraining) %in% 
                         c('classe','user_name', 'cvtd_timestamp', 'new_window'))], 
                       method='pca', thresh=0.9)

preProc

# Building tree model
if (!exists('treeModel')) {
  treeModel <- train( classe ~ ., method='rpart', preProcess='pca', data=pmlTraining)
  treeModel
}

if (!exists('rfModel')) {
  # Since the data is large enough to train well our model, the number of folds can be reduced to a
  # small number, for instance, 5 (actually, I've tried with no folds and works very good as well)
  # The number of trees can also be lowered to 50 due to the amount of variables predicting
  rfModel <- train( classe ~ ., method='rf', preProcess='pca', data=pmlTraining, prox=TRUE, ntree=50,
                    trControl = trainControl(number=5, repeats=1))
  rfModel
}

if (!exists('nbModel')) {
  nbModel <- train( classe ~ ., method='nb', preProcess='pca', data=pmlTraining,
                    trControl = trainControl(number=5, repeats=1))
  nbModel
}

# Confusion matrix
if (exists('rfModel')) {
  resRF <- predict(rfModel, pmlTest)
  table(resRF, pmlTest$classe)
}

# Confusion matrix against NB
if (exists('nbModel') & exists('resRF') ) {
  resNB <- predict(nbModel, pmlTest)
  table(resRF, resNB)
}

submitTest <- read.csv('pml-testing.csv', 
                       header=TRUE, 
                       na.strings=c("","#DIV/0!","NA")
                       )

# Random forest seems to be the most accurate model, so let's use it for the submission
submitResult <- predict(rfModel, submitTest)

pml_write_files(submitResult)
