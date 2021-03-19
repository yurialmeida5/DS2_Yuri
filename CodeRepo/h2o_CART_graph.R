# Load necessary packages and initiate h2o session 

library(tidyverse)
library(h2o)
library(DiagrammeR)
library(rpart)
library(caret)

h2o.init()


# Clear environment and load the data 

rm(list = ls())
data <- as_tibble(ISLR::OJ)
skimr::skim(data)

# Train a decision tree as a benchmark model. 
# Plot the final model and interpret the result (using rpart and rpart.plot is an easier option).

# Create a training data of 75% and keep 25% of the data as a test set. 
my_seed <- 15032021
data_split <- h2o.splitFrame(as.h2o(data), ratios = 0.75, seed = my_seed)
data_train <- data_split[[1]]
data_test <- data_split[[2]]

y <- "Purchase"
x <- setdiff(names(data_train), y)

# mtries: Use this option to specify the number of columns to randomly select at each level.
# sample_rate: This option is used to specify the row (x-axis) sampling rate (without replacement). The range is 0.0 to 1.0. Row and column sampling (sample_rate and col_sample_rate) can improve generalization and lead to lower validation and test set errors. 
# Good general values for large datasets are around 0.7 to 0.8 for both parameters, as higher values generally improve training accuracy.
# max_depth: In general, deeper trees can seem to provide better accuracy on a training set because deeper trees can overfit your model to your data. This is especially true at depths greater than 10. 

# Find the optimal max depth where the AUC is maximized

# GBM hyperparamters
gbm_params = list(max_depth = seq(2, 10))

# Train and validate a cartesian grid of GBMs
gbm_grid = h2o.grid("gbm", x = x, y = y,
                    grid_id = "gbm_grid_1tree8",
                    training_frame = data_train,
                    validation_frame = data_test,
                    ntrees = 1, min_rows = 1, sample_rate = 1, col_sample_rate = 1,
                    learn_rate = .01, seed = my_seed,
                    hyper_params = gbm_params)

gbm_gridperf = h2o.getGrid(grid_id = "gbm_grid_1tree8",
                           sort_by = "auc",
                           decreasing = TRUE)

opt_max_depth <- as.numeric(gbm_gridperf@summary_table$max_depth[1])


simple_tree <- h2o.gbm(x = x, y = y, 
                        training_frame = data_train, 
                        ntrees = 1, min_rows = 1, sample_rate = 1, col_sample_rate = 1,
                        max_depth = opt_max_depth,
                        # use early stopping once the validation AUC doesn't improve by at least 0.01%
                        stopping_rounds = 5, stopping_tolerance = 0.01, 
                        stopping_metric = "AUC", 
                        seed = my_seed)

simpleH2oTree = h2o.getModelTree(model = simple_tree, tree_number = 1)

# Load graph functions and plot the h2o version of the model 
source('HW1/functions.R')

simpleDataTree = createDataTree(simpleH2oTree)

GetEdgeLabel <- function(node) {return (node$edgeLabel)}
GetNodeShape <- function(node) {switch(node$type, split = "diamond", leaf = "oval")}
GetFontName <- function(node) {switch(node$type, split = 'Palatino-bold', leaf = 'Palatino')}
SetEdgeStyle(simpleDataTree, fontname = 'Palatino-italic', label = GetEdgeLabel, labelfloat = TRUE,
             fontsize = "26", fontcolor='royalblue4')
SetNodeStyle(simpleDataTree, fontname = GetFontName, shape = GetNodeShape, 
             fontsize = "26", fontcolor='royalblue4',
             height="0.75", width="1")

SetGraphStyle(simpleDataTree, rankdir = "LR", dpi=70.)

plot(simpleDataTree, output = "graph")

# Generates the rpart version of it, easier to interpret

set.seed(my_seed)
graph_simple_tree <- rpart(Purchase ~ ., data = as.data.frame(data_train), method = "class", 
                           control = rpart.control(cp = 0.01 , minsplit = 1, 
                                                   maxdepth = opt_max_depth))
par(xpd = NA) # Avoid clipping the text in some device
plot(graph_simple_tree)
text(graph_simple_tree, digits = 3)


# Investigate tree ensemble models: random forest, gradient boosting machine, XGBoost. Try various tuning parameter combinations and select the best model using cross-validation.

## h2o was taking to long to generate 

## Random forest

