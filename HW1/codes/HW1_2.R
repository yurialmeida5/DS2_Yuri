# Clear memory
rm(list=ls())

# Libraries

library(tidyverse)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(xgboost)
library(randomForest)
library(caret)
theme_set(theme_minimal())

library(h2o)
h2o.init()
h2o.no_progress()
#h2o.shutdown()

my_seed <- (5678)


# Load the data

data <- as_tibble(ISLR::Hitters) %>%
  drop_na(Salary) %>%
  mutate(log_salary = log(Salary), Salary = NULL)
h2o_data <- as.h2o(data)


# A, Train 2 Random Forest models

y <- "log_salary"
X <- setdiff(names(h2o_data), y)

## 1, Random forest with 2 variables

rf_2_var <- h2o.randomForest(
  X, y,
  training_frame = h2o_data,
  model_id = "rf_2_var",
  mtries = 2,
  seed = my_seed
)

## 2, Random forest with 10 variables

rf_10_var <- h2o.randomForest(
  X, y,
  training_frame = h2o_data,
  model_id = "rf_2_var",
  mtries = 10,
  seed = my_seed
)

## Comparison of variable importance

h2o.varimp_plot(rf_2_var)
h2o.varimp_plot(rf_10_var)

# Comparing the results of the models selecting 2 and 10 variables for each split, the features showed to be quite similar 
# The variables CatBat and CHits demonstrated to be the most important variables in both models.
# However, it's interesting to highlight that there a huge difference in scale of importance of those 2 variables in those 2 models.

# B, Explanation of extreme difference in variable importance

# In our Random Forest Classifier, at each node of a decision tree, the features to be used for splitting the dataset is decided based on information gain(I.G.) 
# or the more computationally cheap Gini impurity reduction. 
# The CAtBat proved to be important for both models with 2 and 10 mtries,
# however with 10 randomly picked variables, the variable will be randomly selected more times by the trees (Default 50 trees)
# With that having its importance "increased in terms of scale.


# C, Two GBM models 

## GBM sample_rate = 0.1

gbm_srate_01 <- h2o.gbm(
  x = X, y = y,
  model_id = "gbm_srate_01",
  training_frame = h2o_data,
  sample_rate = 0.1,
  seed = my_seed
)

## GBM sample_rate = 1

gbm_srate_1 <- h2o.gbm(
  x = X, y = y,
  model_id = "gbm_srate_1",
  training_frame = h2o_data,
  sample_rate = 1,
  seed = my_seed
)

## Comparison of variable importance

h2o.varimp_plot(gbm_srate_01)
h2o.varimp_plot(gbm_srate_1)



# The default sample rate value is 1, which means the bootstrap sample will have the same number of rows as the original data table. 
# If you choose a value that is less than 1, then the bootstrap sample will have fewer rows that in the original table. 
# As the bootstrapping always completely resampled the data building unrelated trees and with a sample_rate equal to 0.1,
# only 10% of the data gets replaced making the generated datasets look similar to each other, "enhancing"  in that way the other variables importance scale.

# Close h2o session
h2o.shutdown()
