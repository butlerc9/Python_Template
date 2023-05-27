# TODO
# - write test feed forward for activation layer
# - feed forward for activation layer
# - write test back prop for activation layer
# - back prop for activation layer
# - write test feed forward for fully connected
# - feed forward for fully connected
# - write test back prop for fully connected

import numpy as np

import pandas as pd

from network_classes.network import network
from network_classes.fully_connected import fully_connected_layer
from network_classes.activation_layer import activation_layer
from helper_functions.activation_funcs import sigmoid, der_sigmoid

df = pd.read_csv('data/raw/mnist_train.csv', nrows=1000)
y_train = df['label'].to_numpy()
x_train = df.drop('label',axis=1).to_numpy()

n_values = np.max(9) + 1
y_train = np.eye(n_values)[y_train]

# transpose operations require a 2d array 
# e.g. a normal row vector (12,) transposed still gives (12,) whereas a (12,1 vector) transposed gives (1,12)
y_train = np.expand_dims(y_train,axis=1)
x_train = np.expand_dims(x_train,axis=2)

net = network()
net.add_layer(fully_connected_layer(784, 10))
net.add_layer(activation_layer(sigmoid, der_sigmoid))
net.add_layer(fully_connected_layer(10, 10))
net.add_layer(activation_layer(sigmoid, der_sigmoid))

# train network
net.train(x_train, y_train, 0.1)

