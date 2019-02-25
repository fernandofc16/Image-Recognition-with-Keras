library(keras)
library(class)

#function min max to normalize values from range [0-255] to [0-1]
rescale <- function(x) (x-min(x))/(max(x) - min(x))

#function to compress colored image into grayscale
toGray <- function(x) {
  return((x[,,,1] + x[,,,2] + x[,,,3]) / 3)
}

#load datasets
cifar <- dataset_cifar10()
mnist <- dataset_mnist()

#get training samples and rescale them
train_x_cifar <- rescale(cifar$train$x)
train_x_mnist <- rescale(mnist$train$x)

#get training labels and convert from numeric to a vector
train_y_cifar <- cifar$train$y
train_y_cifar_v <- matrix(data = 0, nrow=dim(train_x_cifar)[1], ncol=10)
for(i in 1:dim(train_x_cifar)[1]) {
  idx <- train_y_cifar[i]+1
  train_y_cifar_v[i,idx] <- 1
}

train_y_mnist <- mnist$train$y
train_y_mnist_v <- matrix(data = 0, nrow=dim(train_x_mnist)[1], ncol=10)
for(i in 1:dim(train_x_mnist)[1]) {
  idx <- train_y_mnist[i]+1
  train_y_mnist_v[i,idx] <- 1
}

#get testing samples and rescale them
test_x_cifar <- rescale(cifar$test$x)
test_x_mnist <- rescale(mnist$test$x)

#get testing labels and convert from numeric to a vector
test_y_cifar <- cifar$test$y
test_y_cifar_v <- matrix(data = 0, nrow=dim(test_y_cifar)[1], ncol=10)
for(i in 1:dim(test_y_cifar)[1]) {
  idx <- test_y_cifar[i]+1
  test_y_cifar_v[i,idx] <- 1
}

test_y_mnist <- mnist$test$y
test_y_mnist_v <- matrix(data = 0, nrow=dim(test_y_mnist)[1], ncol=10)
for(i in 1:dim(test_y_mnist)[1]) {
  idx <- test_y_mnist[i]+1
  test_y_mnist_v[i,idx] <- 1
}


###  MLP - Multi Layer Perceptron
# For using in MLP the cifar10 dataset will be compressed from 3 depth to 1
# And the matrix of each image sample will be flatten into a vector
train_x_cifar_vector = toGray(train_x_cifar)
dim(train_x_cifar_vector) = c(dim(train_x_cifar_vector)[1], dim(train_x_cifar_vector)[2]*dim(train_x_cifar_vector)[3])

train_x_mnist_vector = train_x_mnist
dim(train_x_mnist_vector) = c(dim(train_x_mnist_vector)[1], dim(train_x_mnist_vector)[2]*dim(train_x_mnist_vector)[3])


#The same will happen for the testing samples
test_x_cifar_vector = toGray(test_x_cifar)
dim(test_x_cifar_vector) = c(dim(test_x_cifar_vector)[1], dim(test_x_cifar_vector)[2]*dim(test_x_cifar_vector)[3])

test_x_mnist_vector = test_x_mnist
dim(test_x_mnist_vector) = c(dim(test_x_mnist_vector)[1], dim(test_x_mnist_vector)[2]*dim(test_x_mnist_vector)[3])

#Create a sequential model for cifar10 dataset
model <- keras_model_sequential()

# define and compile the model
model %>% 
  layer_dense(units = 512, activation = 'relu', input_shape = c(dim(train_x_cifar_vector)[2])) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 10, activation = 'softmax') %>% 
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_sgd(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = TRUE),
    metrics = c('accuracy')     
  )

# train
model %>% fit(train_x_cifar_vector, train_y_cifar_v, epochs = 20, batch_size = 128)

# evaluate
score <- model %>% evaluate(test_x_cifar_vector, test_y_cifar_v, batch_size = 128)
score

# Loss - 1.69
# Accuracy - 40.37%
###################################################################################################################################

#Create a sequential model for mnist dataset
model <- keras_model_sequential()

#define and compile model
model %>% 
  layer_dense(units = 512, activation = 'relu', input_shape = c(dim(train_x_mnist_vector)[2])) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 10, activation = 'softmax') %>% 
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_sgd(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = TRUE),
    metrics = c('accuracy')
  )

# train
model %>% fit(train_x_mnist_vector, train_y_mnist_v, epochs = 20, batch_size = 128)

# evaluate
score <- model %>% evaluate(test_x_mnist_vector, test_y_mnist_v, batch_size = 128)
score

# Loss - 0.061
# Accuracy - 98.18%
###################################################################################################################################

# CNN
img_processed_size = 32
  
#Create a sequential model for cifar10 dataset
model <- keras_model_sequential() 

model %>%  
  #defining a 2-D convolution layer
  layer_conv_2d(filter = img_processed_size, kernel_size = c(3,3), padding = "same", input_shape = c(img_processed_size, img_processed_size, 3)) %>%  
  layer_activation("relu") %>%  
  #another 2-D convolution layer
  layer_conv_2d(filter = img_processed_size, kernel_size=c(3,3))  %>%  layer_activation("relu") %>%
  #Defining a Pooling layer which reduces the dimentions of the features map and reduces the computational complexity of the model
  layer_max_pooling_2d(pool_size=c(2,2)) %>%  
  #dropout layer to avoid overfitting
  layer_dropout(0.25) %>%
  #repeat the process
  layer_conv_2d(filter = img_processed_size, kernel_size=c(3,3), padding = "same") %>% layer_activation("relu") %>%  
  layer_conv_2d(filter = img_processed_size, kernel_size=c(3,3), padding = "same" ) %>%  layer_activation("relu") %>%  
  layer_max_pooling_2d(pool_size=c(2,2)) %>%  
  layer_dropout(0.25) %>%
  #flatten the input  
  layer_flatten() %>%
  #create a dense net to classify
  layer_dense(512) %>%  
  layer_activation("relu") %>%  
  layer_dropout(0.5) %>%  
  #output layer-10 classes-10 units  
  layer_dense(10) %>%  
  #applying softmax nonlinear activation function to the output layer to calculate cross-entropy  
  layer_activation("softmax") 

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

model %>% fit(
  x = train_x_cifar, y = train_y_cifar_v, 
  epochs = 90, batch_size = 128, 
  validation_split = 0.2
)

result <- model %>% evaluate(test_x_cifar, test_y_cifar_v)
result

# Loss - 0.74
# Accuracy - 78.57%
##################################################################################################################

# CNN
img_processed_size = 28

#Create a sequential model for mnist dataset
model <- keras_model_sequential() 

model %>%  
  #defining a 2-D convolution layer
  layer_conv_2d(filter = img_processed_size, kernel_size = c(3,3), padding = "same", input_shape = c(img_processed_size, img_processed_size, 1)) %>%  
  layer_activation("relu") %>%  
  #another 2-D convolution layer
  layer_conv_2d(filter = img_processed_size, kernel_size=c(3,3))  %>%  layer_activation("relu") %>%
  #Defining a Pooling layer which reduces the dimentions of the features map and reduces the computational complexity of the model
  layer_max_pooling_2d(pool_size=c(2,2)) %>%  
  #dropout layer to avoid overfitting
  layer_dropout(0.25) %>%
  #repeat the process
  layer_conv_2d(filter = img_processed_size, kernel_size=c(3,3), padding = "same") %>% layer_activation("relu") %>%  
  layer_conv_2d(filter = img_processed_size, kernel_size=c(3,3), padding = "same" ) %>%  layer_activation("relu") %>%  
  layer_max_pooling_2d(pool_size=c(2,2)) %>%  
  layer_dropout(0.25) %>%
  #flatten the input  
  layer_flatten() %>%
  #create a dense net to classify
  layer_dense(512) %>%  
  layer_activation("relu") %>%  
  layer_dropout(0.5) %>%  
  #output layer-10 classes-10 units  
  layer_dense(10) %>%  
  #applying softmax nonlinear activation function to the output layer to calculate cross-entropy  
  layer_activation("softmax") 

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

#Add depth to dataset mnist of 1
dim(train_x_mnist) = c(dim(train_x_mnist)[1], dim(train_x_mnist)[2], dim(train_x_mnist)[3], 1)
dim(test_x_mnist) = c(dim(test_x_mnist)[1], dim(test_x_mnist)[2], dim(test_x_mnist)[3], 1)

model %>% fit(
  x = train_x_mnist, y = train_y_mnist_v, 
  epochs = 90, batch_size = 128, 
  validation_split = 0.2
)

result <- model %>% evaluate(test_x_mnist, test_y_mnist_v)
result

#Loss - 0.023
#Accuracy - 99.48%
#############################################################################################################################

#cifar10 dataset
#Make the predictions with kNN algorithm, considering the 5 closest training samples, k =5
x <- knn(train = train_x_cifar_vector, test = test_x_cifar_vector, cl = train_y_cifar, k = 5)

#conta-se a quantidade de acertos das predições realizadas pelo algoritmo de KNN
correct <- 0
for(i in 1:dim(test_y_cifar)[1]) {
  if(x[i] == test_y_cifar[i]) {
    correct <- correct + 1
  }
}

#calcula-se a porcentagem de acertos
print(paste('Porcentagem de acertos:', (correct/10000)*100, '%'))

##############################################################################################################################

#mnist dataset
#Make the predictions with kNN algorithm, considering the 5 closest training samples, k =5
x <- knn(train = train_x_mnist_vector, test = test_x_mnist_vector, cl = train_y_mnist, k = 5)

#conta-se a quantidade de acertos das predições realizadas pelo algoritmo de KNN
correct <- 0
for(i in 1:dim(test_y_mnist)[1]) {
  if(x[i] == test_y_mnist[i]) {
    correct <- correct + 1
  }
}

#calcula-se a porcentagem de acertos
print(paste('Porcentagem de acertos:', (correct/10000)*100, '%'))