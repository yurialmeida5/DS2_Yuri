
# Clear memory
rm(list = ls())

# Library

library(tidyverse)
library(data.table)
library(ggplot2)
library(caret)
library(knitr)
library(lubridate)
library(plyr)
library(party)
library(doMC)
library(pROC)
library(ranger)
library(psych)
library(keras)

# Set seed

my_seed <- 28081992

# Data Check: Missing Values / Explanatory Variables Distribution 

data_train <- read_csv("D:/CEU/Data_Science_2/DS2_Yuri/Kaggle_Competition/data/train.csv")
data_test <- read_csv("D:/CEU/Data_Science_2/DS2_Yuri/Kaggle_Competition/data/test.csv")

# Check data structure and distribuitions
skimr::skim(data_train)
str(data_train)
ggplot(gather(data_train, cols, value), aes(value)) +
  geom_histogram() +
  facet_wrap(.~cols, scales = "free")

# No missing values from any of the columns, what is really good for better accuracy of the model predictions. 
# All the 60 columns are numeric ones with a few that could be transformed into factors.




# Some outliers that need to be eliminated
data_train <- data_train %>% 
  filter(n_unique_tokens <=1)

data_train$is_popular <- factor(data_train$is_popular, level = c(1,0), labels = c("Yes","No"))

# Remove article_id from the predictions
data_train$article_id <- NULL

# create Function to submit the files

submit_files_caret <- function(model){
  
  to_submit <- data.table(
    article_id = data_test$article_id,
    score = predict(object = model, newdata = data_test, type = 'prob')[1])
  
  colnames(to_submit) <- c("article_id", "score")
  
  write.csv(to_submit, paste0("D:/CEU/Data_Science_2/DS2_Yuri/Kaggle_Competition/submissions/",substitute(model),".csv"), row.names=FALSE)
  
  save(model, file = paste0("D:/CEU/Data_Science_2/DS2_Yuri/Kaggle_Competition/models/",substitute(model),".RData"))
  
}

# 1st Model - Elastic Net

fitControl <- trainControl(method = "repeatedcv", 
                           number = 5,
                           repeats = 5,
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary,
                           savePredictions = FALSE)

elastic_net <- train(
  is_popular~.,
  data = data_train,
  method="glmnet",
  preProcess=c("center", "scale"),
  trControl=fitControl,
  metric = 'ROC',
  seed = my_seed
)

submit_files(elastic_net)

# 2nd Model - Elastic Net with PCA


fitControl <- trainControl(method = "repeatedcv", 
                                        number = 10,
                                      repeats = 10,
                                  classProbs = TRUE, 
                  summaryFunction = twoClassSummary,
                            savePredictions = FALSE)

elastic_net_pca <- train(
       is_popular~.,
       data = data_train,
       method="glmnet",
       preProcess=c("center", "scale", "pca"),
       trControl=fitControl,
       metric = 'ROC',
       seed = my_seed)

submit_files_caret(elastic_net_pca)

# Random Forest

train_control <- trainControl(
  method = "cv",
  n = 5,
  classProbs = TRUE, 
  summaryFunction = twoClassSummary,
  savePredictions = FALSE
)

tune_grid <- expand.grid(
  .mtry = c(2,3,4,5),
  .splitrule = "gini",
  .min.node.size = c(5, 10, 20)
)

random_forest <- train(is_popular~ .,
                  data = data_train,
                  method = "ranger",
                  trControl = train_control,
                  tuneGrid = tune_grid,
                  # The ’impurity’ measure is the Gini index for classification
                  importance = "impurity",
                  seed = my_seed
)

submit_files_caret(random_forest)

#######################################################################################

## THE CHAMP

# XGBoost

train_control <- trainControl(
  method = "cv",
  n = 10,
  classProbs = TRUE, 
  summaryFunction = twoClassSummary,
  savePredictions = FALSE
)


xgb_grid <- expand.grid(nrounds = 700,
                        max_depth = c(2,4,5,7,9),
                        eta = c(0.005, 0.01),
                        gamma = 0.001,
                        colsample_bytree = c(0.3, 0.5, 0.7),
                        min_child_weight = 1, # similar to n.minobsinnode
                        subsample = c(0.8))

xgboost_champ <- train(is_popular ~ .,
                     method = "xgbTree",
                     data = data_train,
                     trControl = train_control,
                     tuneGrid = xgb_grid,
                     seed = my_seed)

submit_files_caret(xgboost_champ)


# XGBoost - Higher number of 

train_control <- trainControl(
  method = "cv",
  n = 7,
  classProbs = TRUE, 
  summaryFunction = twoClassSummary,
  savePredictions = FALSE
)


xgb_grid <- expand.grid(nrounds = 900,
                        max_depth = c(2,4,5,7,9),
                        eta = c(0.005, 0.01),
                        gamma = 0.001,
                        colsample_bytree = c(0.3, 0.5, 0.7),
                        min_child_weight = 1, # similar to n.minobsinnode
                        subsample = c(0.7))

xgboost_champ2 <- train(is_popular ~ .,
                       method = "xgbTree",
                       data = data_train,
                       trControl = train_control,
                       tuneGrid = xgb_grid,
                       seed = my_seed)

submit_files_caret(xgboost_champ2)


########################## NEURAL NETWORKS ################################

x_train <- read_csv("D:/CEU/Data_Science_2/DS2_Yuri/Kaggle_Competition/data/train.csv")
x_valid <- read_csv("D:/CEU/Data_Science_2/DS2_Yuri/Kaggle_Competition/data/test.csv")


submit_files_keras <- function(model){
  
  to_submit <- data.table(article_id = y_valid,
                          preds = model %>% predict_proba(x_valid))
  
  to_submit <- to_submit[, c(1,3)]
  
  colnames(to_submit) <- c("article_id", "score")
  
  write.csv(to_submit, paste0("D:/CEU/Data_Science_2/DS2_Yuri/Kaggle_Competition/submissions/",substitute(model),".csv"), row.names=FALSE)
  
  save(model, file = paste0("D:/CEU/Data_Science_2/DS2_Yuri/Kaggle_Competition/models/",substitute(model),".RData"))
  
}


# Scale the data before doing anything

y_train <- to_categorical(x_train$is_popular)

# Transform Datasets
x_train <- x_train %>%
                     select(-c("is_popular","article_id")) %>% 
                     filter(n_unique_tokens <=1) %>% 
                     scale()

y_valid <- x_valid$article_id

x_valid <- x_valid %>%
          select(-c("article_id")) %>% 
          filter(n_unique_tokens <=1) %>% 
          scale()  
   

  
# Neural Network - 1 Relu

set.seed(my_seed)
neuralnet_ReLu <- keras_model_sequential()
neuralnet_ReLu %>%
  layer_dense(units = 300, activation = 'relu', input_shape = ncol(x_train)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 2, activation = 'sigmoid')

compile(
  neuralnet_ReLu,
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

# Add the data

history <- fit(
  neuralnet_ReLu, 
  x_train, y_train,
  epochs = 50,
  batch_size = 250,
  validation_split = 0.5
)


submit_files_keras(neuralnet_ReLu)


# Neural Network - 2 Relus

set.seed(my_seed)
neuralnet_ReLu_2 <- keras_model_sequential()
neuralnet_ReLu_2 %>%
  layer_dense(units = 300, activation = 'relu', input_shape = ncol(x_train)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 300, activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 2, activation = 'sigmoid')

compile(
  neuralnet_ReLu_2,
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

# Add the data

history <- fit(
  neuralnet_ReLu_2, 
  x_train, y_train,
  epochs = 50,
  batch_size = 250,
  validation_split = 0.5
)


submit_files_keras(neuralnet_ReLu_2)




