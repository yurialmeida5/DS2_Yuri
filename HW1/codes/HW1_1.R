# clear memory
rm(list=ls())

#Load necessary packages

library(tidyverse)
library(caret)
library(rpart.plot)
library(gbm)
library(xgboost)
library(caTools)
library(pROC)
library(viridis)
library(skimr)
library(MLeval)
library(h2o)

# Load necessary graph functions
source("HW1/codes/functions.R")

data <- as_tibble(ISLR::OJ)
skimr::skim(data)

# Quick analysis into the data-set

# 2 factor variables, and 16 numeric ones , 1070 observations 
# Non-missing values found (can tangibly reduce prediction power), with that no cleaning steps or imputation methods are necessary.
# Classification models can run smoothly without any data preparation.

# Note: Due to the dataset size, huge amount of time to process the models,
# and the non-possibility to run the XGBOOST model with h2o; 
# for this entire exercise, I've decided to use the package caret.

# Although it's possible to visualize some small differences into the code itself, 
# the hyperparameters can be defined more less in the same way. 


# a. ----------------------------------------------------------------------

# Create a training data of 75% and keep 25% of the data as a test set.

my_seed <- 19920828
set.seed(my_seed)
train_indices <- as.integer(createDataPartition(data$Purchase, 
                                                p = 0.75, list = FALSE))
data_train <- data[train_indices, ]
data_test <- data[-train_indices, ]

# Train a decision tree as a benchmark model.

fitControl <- trainControl(method = "cv", 
                           number = 5,
                           summaryFunction = twoClassSummary,
                           classProbs=TRUE,
                           verboseIter = TRUE,
                           savePredictions = TRUE)

# Hyperparameters: TrainControl

# method = "cv": Single K-fold cross-validation, no bootstrapping ("repeatedcv" re-sampling) needed.
# Dichotomous response variable is balanced in the dataset. (CH and MM - close number of occurences.)
# verboseIter = TRUE: Prints the training log.
# classProbs = TRUE: Show class probabilities.
# savedPredictions = TRUE: Save the hold-out predictions of each resample.
# summaryFunction = twoClassSummary: computes sensitivity, specificity and the area under the ROC curve

# Hyperparameters: CART (Method:  "rpart")

# Caret uses cross-validation to optimize the CART model hyperparameters. 
# The package can call rpart() function and train the model through cross-validation.

# To tune the complexity parameter(cp), set method = "rpart".
# To tune the maximum tree depth, set method = "rpart2"
# tuneLength: Tell how many models should be generated.
# Let it default in this case. (small amount of variable) 


set.seed(my_seed)
model_tree_benchmark<-train(
  Purchase~.,
  data=data_train,
  method="rpart",
  trControl=fitControl
)

model_tree_benchmark

# Using cp or max_depth as tuning parameters generated pretty similar results. 
# However, the cp tunned reduced the complexity of the model with less amount of features (in most cases). 
# Easier to explain.

# Plot the final model and interpret the result (using rpart and rpart.plot is an easier option).

plot(model_tree_benchmark)
evalm(model_tree_benchmark)$stdres
rpart.plot(model_tree_benchmark$finalModel)

# Taking a quick look at the results of the CART model, 
# It's possible to mention that the ROC value achieved a substantial value (higher than 0.8 where the "perfect" model is 1.0),
# With that, the "trade-off" rates demonstrated pretty excellent as well (for the optimal classification-threshold-invariant): 
# Sensitivity (True Positives / All real Positives) is higher than 85% and Specificity (True Negatives / All real Negatives) is also bigger than 70%.

# Now, taking a look into the plotted graph and having those rates into consideration, 
# We can visualized the main important variables in order (LoyalCH , ListPriceDiff, PriceDiff), their decision cutoff's 
# And the distribuition of the dataset under those specific characteristics and it's predict values for the following conditions.
# Ex: If a observation comes up with a LoyalCH lower than 0.48, the model predicts the purchase as "MM"


# b. ----------------------------------------------------------------------

# Investigate tree ensemble models: random forest, gradient boosting machine, XGBoost.
# Try various tuning parameter combinations and select the best model using cross-validation.

## Probability Forest

train_control <- trainControl(
  method = "cv",
  n = 5,
  classProbs = TRUE, 
  summaryFunction = twoClassSummary,
  savePredictions = TRUE,
  verboseIter = TRUE
)

tune_grid <- expand.grid(
  .mtry = c(2,3,4,5),
  .splitrule = "gini",
  .min.node.size = c(5, 10, 20)
)

# Hyperparameters: Random Forest (Method: ranger)

# mtry: Number of variables randomly sampled as candidates at each split.
# splitrule: 'Gini'  
# The ’impurity’ measure is the Gini index for classification, the variance of the responses for regression and the sum of test statistics
# min.node.size: Minimal node size. Default 1 for classification, 5 for regression, 3 for survival, and 10 for probability

# random forest
set.seed(my_seed)
model_rf <- train(Purchase~ .,
                  data = data_train,
                  method = "ranger",
                  trControl = train_control,
                  tuneGrid = tune_grid,
                  # The ’impurity’ measure is the Gini index for classification
                  importance = "impurity"
)

model_rf
# Best model mtry = 5, splitrule = gini and min.node.size = 20, ROC = 0.89
evalm(model_rf)$stdres


## Gradient Boosting Machine

gbmGrid <-  expand.grid(interaction.depth = c(1, 3, 5, 7, 9), 
                        n.trees = 500, # high number of trees, small shrinkage
                        shrinkage = c(0.005, 0.01),
                        n.minobsinnode = c(2,3,4,5)) 

# Hyperparameters: Stochastic Gradient Boosting (Method: gbm)

# interaction.depth (Maximum nodes per tree) - number of splits it has to perform on a tree (starting from a single node).
# n.trees: (number of trees) Increasing N reduces the error on training set, but setting it too high may lead to over-fitting.
# shrinkage: Learning Rate - High Values (close to 1.0) poor performance (over-fitting), Small-ones (lower than 0.01) requires big n.trees
# n.minobsinnode: the minimum number of observations in trees' terminal nodes. Small samples - lower this setting to five or even three.

set.seed(my_seed)
model_gbm <- train(Purchase~ .,
                   data = data_train,
                   method = "gbm",
                   trControl = train_control,
                   tuneGrid = gbmGrid,
                   # will show you both WARNING and INFO log levels
                   verbose = TRUE
)
model_gbm 
# Best model n.trees = 500, interaction.depth = 3, shrinkage = 0.005 and n.minobsinnode = 3, ROC = 0.90
evalm(model_gbm)$stdres

## XGBoost

# Hyperparameters: Stochastic Gradient Boosting (Method: gbm)

# nrounds: max number of boosting iterations
# max_depth: maximum depth of a tree
# eta: control the learning rate (Lower value / Larger value for nrounds - Less overfitting): Default: 0.3 
# gamma: minimum loss reduction required to make a further partition on a leaf node of the tree. (Higher Value, + conservative)
# colsample_bytree: subsample ratio of columns when constructing each tree.
# min_child_weight: minimum sum of instance weight (hessian) needed in a child. (Higher Value, + conservative)
# subsample: subsample ratio of the training instance. 
# Setting it to 0.5 means that xgboost randomly collected half of the data instances to grow trees and this will prevent overfitting.


xgb_grid <- expand.grid(nrounds = 500,
                        max_depth = c(1, 3, 5, 7, 9),
                        eta = c(0.005, 0.01),
                        gamma = 0.01,
                        colsample_bytree = c(0.3, 0.5, 0.7),
                        min_child_weight = 1, # similar to n.minobsinnode
                        subsample = c(0.5))
set.seed(my_seed)
model_xgboost <- train(Purchase ~ .,
                       method = "xgbTree",
                       data = data_train,
                       trControl = train_control,
                       tuneGrid = xgb_grid)
model_xgboost
# Best model were nrounds = 500, max_depth = 3, eta = 0.005, gamma = 0.01, colsample_bytree = 0.7, min_child_weight = 1, ROC = 0.9
evalm(model_gbm)$stdres


# c. ----------------------------------------------------------------------

# Compare the performance of the different models (if you use caret you should consider using the resamples function). 

resamples <- resamples(list("decision_tree_benchmark" = model_tree_benchmark,
                            "rf" = model_rf,
                            "gbm" = model_gbm,
                            "xgboost" = model_xgboost))
summary(resamples)

logit_models <- list()
logit_models[["decision_tree_benchmark"]] <- model_tree_benchmark
logit_models[["RF"]] <- model_rf
logit_models[["GBM"]] <- model_gbm
logit_models[["XGBoost"]] <- model_xgboost

CV_AUC_folds <- list()

for (model_name in names(logit_models)) {
  
  auc <- list()
  model <- logit_models[[model_name]]
  for (fold in c("Fold1", "Fold2", "Fold3", "Fold4", "Fold5")) {
    cv_fold <-
      model$pred %>%
      filter(Resample == fold)
    
    roc_obj <- roc(cv_fold$obs, cv_fold$CH)
    auc[[fold]] <- as.numeric(roc_obj$auc)
  }
  
  CV_AUC_folds[[model_name]] <- data.frame("Resample" = names(auc),
                                           "AUC" = unlist(auc))
}

CV_AUC <- list()

for (model_name in names(logit_models)) {
  CV_AUC[[model_name]] <- mean(CV_AUC_folds[[model_name]]$AUC)
}

unlist(CV_AUC)

# Is any of these giving significantly different predictive power than the others?

# Yes, although the CART benchmark model is the easiest one to interpret, 
# its AUC (for the optimal classification-threshold-invariant) results showed significantly lower than the other models (RF, GBM XGboost)
# with a slightly advantage to gbm.

# d. ----------------------------------------------------------------------

# Choose the best model and plot ROC curve for the best model on the test set

## ROC Plot with built-in package 
gbm_pred <-predict(model_gbm, data_test, type="prob")
colAUC(gbm_pred, data_test$Purchase, plotROC = TRUE)

## ROC plot with own function
data_test[,"best_model_no_loss_pred"] <- gbm_pred[,"CH"]

roc_obj_holdout <- roc(data_test$Purchase, data_test$best_model_no_loss_pred)

createRocPlot <- function(r, plot_name) {
  all_coords <- coords(r, x="all", ret="all", transpose = FALSE)
  
  roc_plot <- ggplot(data = all_coords, aes(x = fpr, y = tpr)) +
    geom_line(color='blue', size = 0.7) +
    geom_area(aes(fill = 'red', alpha=0.4), alpha = 0.3, position = 'identity', color = 'blue') +
    scale_fill_viridis(discrete = TRUE, begin=0.6, alpha=0.5, guide = FALSE) +
    xlab("False Positive Rate (1-Specifity)") +
    ylab("True Positive Rate (Sensitivity)") +
    geom_abline(intercept = 0, slope = 1,  linetype = "dotted", col = "black") +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, .1), expand = c(0, 0.01)) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .1), expand = c(0.01, 0)) + 
    theme_bw()
  
  roc_plot
}

createRocPlot(roc_obj_holdout, "ROC curve for best model (GBM)")

# AUC provides an aggregate measure of performance across all possible classification thresholds. 
# A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.
# In our case, the gbm model for the test set resulted in the value of 0.8733, 
# what is a really good indication that our model performed really great predictions in the data test set. 
# The graph bellow shows the AUC threshold trade-offs, where you can determine the TP rate and the FP rate for all the points inside the blue line.


# e. ----------------------------------------------------------------------

# Inspect variable importance plots for the 3 models.

plot(varImp(model_rf), top = 5)

plot(varImp(model_gbm), top = 5)

plot(varImp(model_xgboost), top = 5)

# Yes, the variables ended up being more less similar important for the RF, GBM and XGboost models.
# What is interesting to highlight are specially the variables (LoyalCH and PriceDiff), 
# They have showed to be the first and second most important variables in all  of the 3 models, 
# compounding the biggest part of the predictions contribution.