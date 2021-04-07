# clear memories
rm(list = ls())

# Load libraries

library(tidyverse)
library(keras)
library(grid)

# install.packages("tensorflow")
library(tensorflow)
# install_tensorflow()
# library(reticulate)
# install_miniconda()

my_seed <- 28081992

# Exercise 1 - Fashion MNIST data

# The MNIST database is a large database of handwritten digits that is commonly used for training various image processing systems. 
# The black and white images from NIST were normalized to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels.
# The MNIST database contains 60,000 training images and 10,000 testing images

## Load data

fashion_mnist <- dataset_fashion_mnist()
x_train <- fashion_mnist$train$x
y_train <- fashion_mnist$train$y
x_test <- fashion_mnist$test$x
y_test <- fashion_mnist$test$y

class_names <- c('T-shirt',
                 'Trouser',
                 'Pullover',
                 'Dress',
                 'Coat', 
                 'Sandal',
                 'Shirt',
                 'Sneakers',
                 'Bag',
                 'Boots')

# a) Show some example images from the data

### From the Tension flow documentation

# Divide the plot area into m*n array of subplots
# 2 Lines / 5 Columns
par(mfrow = c(2, 5))

# Mar: A numeric vector of length 4, which sets the margin sizes in the following order: bottom, left, top, and right. 
# The default is c(5.1, 4.1, 4.1, 2.1).
# xaxs = 'i' and yaxs = 'i' : Limits of the images lies on the edges of the array subplots
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')

for (i in 1:10) { 
  img <- x_train[i, , ]
  # Apply the reverse vector on the image matrix columns and transpose 
  # This basically "rotates the image for a better visualization
  img <- t(apply(img, 2, rev)) 
  # Remove the X and Y axis: axes = FALSE / or you could use: xaxt = 'n', yaxt = 'n' 
  image(1:28, 1:28, img, col = gray((0:255)/255), axes = FALSE ,
       # add the clothes names to the title
         main = paste(class_names[y_train[i] + 1]))
}

# Cloth Types / class numbers
print(unique(y_train))

# The for loop above display the first 10 items of the train dataset. 
# The types of the clothes (named here as "classes") are numbered from 0 to 9 
# And define which type of clothes we are dealing with.

## b) Train a fully connected deep network to predict items

# Normalizing the data

# Important observations:

# Data scaling is a recommended pre-processing step when working with deep learning neural networks.
# Data scaling can be achieved by normalizing or standardizing real-valued input and output variables.

# A good rule of thumb is that input variables should be small values, 
# probably in the range of 0-1 or standardized with a zero mean and a standard deviation of one

#### Scaling the axis so that they are between 0 and 1 

x_train <- array_reshape(x_train, c(dim(x_train)[1], 784)) 
x_test <- array_reshape(x_test, c(dim(x_test)[1], 784)) 

x_train <- as.matrix(x_train) / 255
x_test <- as.matrix(x_test) / 255

####  One-hot encoding for the outcome

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

## Building the first model

# Activation Functions: 

## Linear: The output is same as the input and the function is defined in the range (-infinity, +infinity).
## Unit Step: The output assumes value 0 for negative argument and 1 for positive argument. The range is between (0,1) and the output is binary in nature.
## Sigmoid: This squashes the input to any value between 0 and 1, and makes the model logistic in nature.
## Tanh: This is a nonlinear function, defined in the range of values (-1, 1).The gradient is stronger for tanh than sigmoid.
## ReLU: The range of output is between 0 and infinity. 

set.seed(my_seed)
base_model <- keras_model_sequential()
base_model %>%
  # The input_shape argument to the first layer specifies the shape of the input data (a length 784 numeric vector representing a grayscale image).
  # The input shape is the only one you must define, because your model cannot know it. Only you know that, based on your training data.
  # All the other shapes are calculated automatically based on the units and particularities of each layer.
  layer_dense(units = 400, activation = 'relu', input_shape = c(784)) %>% # adding 128 nods
  # Dropout is a technique where randomly selected neurons are ignored during training. 
  # They are “dropped-out” randomly. This means that their contribution to the activation of downstream neurons is temporally removed 
  # on the forward pass and any weight updates are not applied to the neuron on the backward pass.
  # The effect is that the network becomes less sensitive to the specific weights of neurons. 
  # This in turn results in a network that is capable of better generalization and is less likely to overfit the training data.
  layer_dropout(rate = 0.4) %>% 
  # Softmax is generalization of the logistic function to multiple dimensions.
  # Multinomial logistic regression, often used as the last activation function to normalize the output of a network 
  # to a probability distribution over predicted output classes
  layer_dense(units = 10, activation = 'softmax') # 10 nods since 10 results

compile(
  base_model,
  # Use this crossentropy loss function when there are two or more label classes. 
  # Expected labels to be provided in a one_hot representation.
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# Adding the data

history_1 <- fit(
  base_model, x_train, y_train,
  # Batch_Size: 
  # The batch size defines the number of samples that will be propagated through the network.
  # It has been observed in practice that when using a larger batch there is a significant degradation in the quality of the model, 
  # as measured by its ability to generalize. (1/10 or 1/8 of the data usually good batch sizes).
  # Epochs: One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
  # It's related to how diverse the dataset is.
  epochs = 30, batch_size = 250,
  # Creates a validation dataset 
  validation_split = 0.3
)

plot(history_1)

# This first model kind does really well on the validation dataset and it's not overfitting the validation dataset. 
# It's compounded by a simple ReLu layer with 400 nods, 30 epochs and batch_size = 250. PS: The number of epochs and batch size will repeat for all the other upcoming models.


# Using the sigmoid activation function

set.seed(my_seed)
sigmoid_model <- keras_model_sequential()
sigmoid_model %>%
  layer_dense(units = 50, activation = 'softmax', input_shape = c(784)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = 'softmax')

compile(
  sigmoid_model,
  # Use this crossentropy loss function when there are two or more label classes. 
  # Expected labels to be provided in a one_hot representation.
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# Add the data

history_2 <- fit(
  sigmoid_model, 
  x_train, y_train,
  epochs = 30,
  batch_size = 250,
  validation_split = 0.3
)

plot(history_2)

# As expected, reducing the number of layers and changing the activation function to sigmoid, 
# diminished drastically the accuracy of our model. 
# The explanation for that is that the derivative of the sigmoid function is always smaller than 1.
# And with many layers (what are really important for deep learning) you will multiply these gradients, and the product of many smaller 
# than 1 values goes to zero very quickly.


# Adding another Relu layer to the neural network

set.seed(my_seed)
relu_model_2 <- keras_model_sequential()
relu_model_2 %>%
  layer_dense(units = 250, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 250, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(relu_model_2)

compile(
  relu_model_2,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_3 <- fit(
  relu_model_2, 
  x_train, y_train,
  epochs = 30,
  batch_size = 250,
  validation_split = 0.3
)

plot(history_3)

# The following model showed pretty closed results to the base_model. 
# However, with this one I would be a little bit more worried about overfitting

# Mixing the activation functions: (Adding different types of layers together - ReLu and SoftMax)

set.seed(my_seed)
mix_model <- keras_model_sequential()
mix_model %>%
  layer_dense(units = 250, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 10, activation = 'softmax', input_shape = c(784)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = 'softmax')

compile(
  mix_model,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_4 <- fit(
  mix_model, 
  x_train, y_train,
  epochs = 30,
  batch_size = 250,
  validation_split = 0.3
)

plot(history_4)

# Adding a sigmoid model helped a lot to reduce the overfitting 
# However it performed really poorly in terms of accuracy comparing to the other models.

# Final Model

# As the model number was performing greatly on the dataset, 
# I have decided to enhance the number of layers units (nodes), reduce the batch size 
# and increase the number of epochs.

set.seed(my_seed)
final_model <- keras_model_sequential()
final_model %>%
  layer_dense(units = 500, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 10, activation = 'softmax')

compile(
  final_model,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_5 <- fit(
  final_model, 
  x_train, y_train,
  epochs = 50,
  batch_size = 200,
  validation_split = 0.3
)

plot(history_5)

# The results obtained were still really good (pretty great accuracy), 
# although the model 5 is way less prone to overfitting.

summary <- data.frame(
  base_model = mean(history_1$metrics$val_accuracy),
  sigmoid_model = mean(history_2$metrics$val_accuracy),
  relu_model_2 = mean(history_3$metrics$val_accuracy),
  mix_model = mean(history_4$metrics$val_accuracy),
  final_model = mean(history_5$metrics$val_accuracy)
)

summary

# The table bellow consolidates the accuracy results of all the neural network models.
# As expected, the final_model was the one that presented the best accuracy and its less prone to overfitting.

# c) Evaluate the model on the test set

evaluate(final_model, x_test, y_test)

# The final accuracy on the test set was really close to 0.9, what is a really good result overall. 
# At the end, it was really good to perceive that the results for the validation and the test set performed really close. 
# The amount of observations between those sets were really similar (Validation - 18k and Test - 10k), assuring some sort of "stability"
# to the model predictions.

# d) Convulated network

# The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. 
# While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.

# In order to use the convulted network method let's reset the datasets and move them back to an array and reset 
# the datasets to a 3D shape to max pooling and 2D convolutional layer processing. 
# After this stage the data is flattened into a 1D shape for the dense network part of all the neural net models.

# Switching the datasets to a 3D shape for pooling a 2D convolutional layer processing. 

x_train <- fashion_mnist$train$x
x_test <- fashion_mnist$test$x

x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))

x_train <- x_train / 255
x_test <- x_test / 255

### Bulding the base model - cnn_base_model

set.seed(my_seed)
cnn_base_model <- keras_model_sequential()
cnn_base_model %>%
  # The kernel size here refers to the widthxheight of the filter mask.
  # Returns the pixel with maximum value from a set of pixels within a mask (kernel). 
  # That kernel is swept across the input, subsampling it.
  layer_conv_2d(
    filters = 32,  # This is closer to what we have seen as the number of nodes 
    kernel_size = c(3, 3), # max the input based on 3x3 squares
    activation = 'relu',
    input_shape = c(28, 28, 1)
  ) %>%
 # Downsamples the input representation by taking the maximum value over the window defined by pool_size for each dimension along the features axis
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% # In this case is downsampling by (2x2)
  layer_dropout(rate = 0.25) %>%
 # Flatten is the function that converts the pooled feature map to a single column (vector) that is passed to the fully connected layer.  
  layer_flatten() %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 10, activation = 'softmax')

compile(
  cnn_base_model,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_6 <- fit(
  cnn_base_model, x_train, y_train,
  epochs = 30, batch_size = 250,
  validation_split = 0.3
)

plot(history_6)


# Our base model in this scenario already shows really great results, 
# however it's important to mention that the model may be overfitting the data set

# Adding an Extra Max Pooling 

set.seed(my_seed)
cnn_extraMax <- keras_model_sequential()
cnn_extraMax %>%
  layer_conv_2d(
    filters = 32,
    kernel_size = c(3, 3),
    activation = 'relu',
    input_shape = c(28, 28, 1)
  ) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 10, activation = 'softmax')

compile(
  cnn_extraMax,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_7 <- fit(
  cnn_extraMax, x_train, y_train,
  epochs = 30, batch_size = 250,
  validation_split = 0.3
)

plot(history_7)

# Adding another max pooling reduced the accuracy of our model, 
# what could be expected considering that the Maxpool removes information from the signal, dropout forces distributed representation, 
# thus both effectively make it harder to propagate information. However, the model itself is a bit less prone to overfitting.

# Adding a Relu before the final output Layer

set.seed(my_seed)
cnn_ReLu <- keras_model_sequential()
cnn_ReLu  %>%
  layer_conv_2d(
    filters = 32,   
    kernel_size = c(3, 3), 
    activation = 'relu',
    input_shape = c(28, 28, 1)
  ) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% # In this case is downsampling by (2x2)
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 10, activation = 'softmax')

compile(
  cnn_ReLu,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_8 <- fit(
  cnn_ReLu, x_train, y_train,
  epochs = 30, batch_size = 250,
  validation_split = 0.3
)

plot(history_8)

# Adding the a ReLu layer before the final output layer slightly improved the accuracy of our model, 
# Although, it's important to mention that the model is still possibly overfitting the dataset 
# based on the results obtained in the validation dataset.

# Adding an extra Max Pooling and a ReLu before the output Layer

set.seed(my_seed)
cnn_extraMax_ReLu <- keras_model_sequential()
cnn_extraMax_ReLu  %>%
  layer_conv_2d(
    filters = 32,   
    kernel_size = c(3, 3), 
    activation = 'relu',
    input_shape = c(28, 28, 1)
  ) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 10, activation = 'softmax')

compile(
  cnn_extraMax_ReLu,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_9 <- fit(
  cnn_extraMax_ReLu, x_train, y_train,
  epochs = 30, batch_size = 250,
  validation_split = 0.3
)

plot(history_9)

# This final model showed results similar to the base model (great accuracy), however due to the addition of the Max Pooling function the model is way less inclinated to overfitting  

# Summary

summary_2 <- data.frame(
  cnn_base_model = mean(history_6$metrics$val_accuracy),
  cnn_extraMax = mean(history_7$metrics$val_accuracy),
  cnn_ReLu = mean(history_8$metrics$val_accuracy),
  cnn_extraMax_ReLu = mean(history_9$metrics$val_accuracy))

summary_2

# The best is actually model cnn_Relu, but we still need to check how it`s performing in the test set.

evaluate(cnn_ReLu, x_test, y_test)

# On the same way that have happened with the neural networks the accuracy on the test set was really close to 0.9, what is a really good result overall. 
# At the end, it was really good to perceive that the results for the validation and the test set performed really close. 
# The amount of observations between those sets were really similar (Validation - 18k and Test - 10k), assuring some sort of "stability"
# to the model predictions.