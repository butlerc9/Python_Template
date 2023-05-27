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

df = pd.read_csv('data/raw/mnist_train.csv',nrows=2)
y_train = df['label'].to_numpy()
x_train = df.drop('label',axis=1).to_numpy()

net = network()
net.add_layer(fully_connected_layer(784, 6))
net.add_layer(fully_connected_layer(6, 5))
net.add_layer(fully_connected_layer(5, 4))
net.add_layer(fully_connected_layer(4, 10))

net.train(x_train, y_train, 0.1)

