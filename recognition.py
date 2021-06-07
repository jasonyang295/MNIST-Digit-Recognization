import numpy as np
import tensorflow as tf 
import cv2 as cv
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize here to scale it down
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

#dont do it for y, only for x


#built model with 2 hidden layers and 1 dense output layer with softmax function

model = tf.keras.models.Sequential() #initalize the grid
model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))#flatten it out and input shape is 28 x 28
#now connect all neurons to previous and following layers with 2 hidden layers and relu activation function
model.add(tf.keras.layers.Dense(units = 128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(units = 128, activation = tf.nn.relu))

#next line scales nums down with softmax function so they add up to 1 so we have actual percentages
model.add(tf.keras.layers.Dense(units = 10, activation = tf.nn.softmax))

#define optimization and loss function as well as 
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 3) #epochs specifies how many times the network trains / sees data


