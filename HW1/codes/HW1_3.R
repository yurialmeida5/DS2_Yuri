
# Clear memory
rm(list=ls())

# Libraries

library(tidyverse)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(xgboost)
library(randomForest)
library(DiagrammeR)
library(plotROC)
theme_set(theme_minimal())

library(h2o)
h2o.init()
#h2o.shutdown()
#h2o.no_progress()


my_seed <- (28081992)

# Load the data

data <- read_csv("HW1/KaggleV2-May-2016.csv")

# some data cleaning
data <- select(data, -one_of(c("PatientId", "AppointmentID", "Neighbourhood"))) %>%
  janitor::clean_names()

# for binary prediction, the target variable must be a factor + generate new variables
data <- mutate(
  data,
  no_show = factor(no_show, levels = c("Yes", "No")),
  handcap = ifelse(handcap > 0, 1, 0),
  across(c(gender, scholarship, hipertension, alcoholism, handcap), factor),
  hours_since_scheduled = as.numeric(appointment_day - scheduled_day)
)

# clean up a little bit
data <- filter(data, between(age, 0, 95), hours_since_scheduled >= 0) %>%
  select(-one_of(c("scheduled_day", "appointment_day", "sms_received")))

h2o_data <- as.h2o(data)

skimr::skim(data)

# 2 factor variables, and 16 numeric ones , 1070 observations 
# Non-missing values found (can tangibly reduce prediction power), with that no cleaning steps or imputation methods are necessary.
# Balanced response value no_show. (51417 No's and 20517 Yes)
# Classification models can run smoothly without any further data preparation.

# A, Create train / validation / test sets, cutting the data into 5% - 45% - 50% parts

splitted_data <- h2o.splitFrame(h2o_data, ratios = c(0.05, 0.45), seed = my_seed)
data_train <- splitted_data[[1]]
data_valid <- splitted_data[[2]]
data_test <- splitted_data[[3]]

# Train rain a benchmark model of your choice and evaluate it on the validation set.

# Glm: Elastic Net (Better Results)
y <- "no_show"
X <- setdiff(names(h2o_data), y)

glm_model <- h2o.glm(
  X, y,
  family = 'binomial',
  training_frame = data_train,
  model_id = "lm",
  nfolds = 5,
  seed = my_seed,
  keep_cross_validation_predictions = TRUE
)

h2o.performance(glm_model, data_valid)

# As the focus of the exercise is based on prediction, the most important metric to evaluate is the AUC (as no_show is balanced).
# that defines how well a binary classification model is able to distinguish between true positives and false positives.
# An AUC of 1 indicates a perfect classifier, while an AUC of .5 indicates a poor classifier, whose performance is no better than random guessing.
# On the following example (Elastid Net Model), you can visualize a really low AUC value (0.58) and the maximum accuracy (the number of correct predictions made as a ratio of all predictions made.)
# that can be obtained as 0.51, this is just slightly better than random predictions.


# C, Build at least 3 models of different families using cross validation, keeping cross validated predictions.
# You might also try deeplearning.

## Random forest

rf_model <- h2o.randomForest(
  X, y,
  training_frame = data_train,
  model_id = "rf",
  ntrees = 200,
  max_depth = 10,
  seed = my_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)


## GBM

gbm_model <- h2o.gbm(
  X, y,
  training_frame = data_train,
  model_id = "gbm",
  ntrees = 200,
  max_depth = 5,
  learn_rate = 0.1,
  seed = my_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)

## Deep learning

deeplearning_model <- h2o.deeplearning(
  X, y,
  training_frame = data_train,
  model_id = "deeplearning",
  hidden = c(32, 8),
  seed = my_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)

# D, Evaluate validation set performance of each model.

# predict on validation set
validation_performances <- list(
  "glm" = h2o.auc(h2o.performance(glm_model, newdata = data_valid)),
  "rf" = h2o.auc(h2o.performance(rf_model, newdata = data_valid)),
  "gbm" = h2o.auc(h2o.performance(gbm_model, newdata = data_valid)),
  "deeplearning" = h2o.auc(h2o.performance(deeplearning_model, newdata = data_valid))
)
# look at AUC for all models
validation_performances

my_models <- list(
  glm_model, gbm_model, rf_model, deeplearning_model)

auc_on_validation <- map_df(my_models, ~{
  tibble(model = .@model_id, AUC = h2o.auc(h2o.performance(., data_valid)))
}) %>% arrange(AUC)

auc_on_validation

# Based on the table analysis, it's possible to visualize that the LM, deeplearning and rf showed pretty similar AUC results. 
# with a slighter advantage to the linear model (the Winner) in this case.
# However, the selection process between them should also be based on the importance rates of identifying FP/FN. 
# As the results were pretty similar it's important to investigate the optimal threshold defined for each one of those models
# based on those importance rates.

# E, How large are the correlations of predicted scores of the validation set produced by the base learners?

h2o.model_correlation_heatmap(my_models, data_valid)

# Based on the heating map, it's clear to perceive a stronger correlation between the deeplearning model with RF and LM.
# With that in consideration, I have decided not to use deeplearning on the final ensemble model.
# The other models demonstrated a correlation lower then 70-80% between them, which means that we can still use them in the final ensemble model.

# F, Create a stacked ensemble model from the base learners.

ensemble_models <- list(glm_model, gbm_model, rf_model)

ensemble_model <- h2o.stackedEnsemble(
  X, y,
  training_frame = data_train,
  metalearner_algorithm = "glm",
  model_id = "stacked_model_glm",
  base_models = ensemble_models,
  seed = my_seed
)


# G, Evaluate ensembles on validation set. Did it improve prediction?

auc_on_validation <- map_df(
  c(my_models, ensemble_model),
  ~{tibble(model = .@model_id, auc = h2o.auc(h2o.performance(., newdata = data_valid)))}
)

auc_on_validation

# Yes, at the end we could have improved slightly bit the previous glm model, however that improvement wasn't significant. 
# The mechanism for improved performance with ensembles is often the reduction in the variance component of prediction errors made by the contributing models.
# As the models generated pretty similar results, it's possible that the benefits contracted from this variance component weren't really notable. 

# H, Evaluate the best performing model on the test set
# How does performance compare to that of the validation set?

final_performance <- tibble(
  "ensemble_model_validation" = h2o.auc(h2o.performance(ensemble_model, newdata = data_valid)),
  "ensemble_model_test" = h2o.auc(h2o.performance(ensemble_model, newdata = data_test)),
)

final_performance

# The results showed pretty much similar to each other (AUC indicators), there wasn't a significant difference in terms of performance between them.
