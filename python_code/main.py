import numpy as np

from helper_functions.activation_funcs import *
from network_classes.activation_layer import activation_layer
from network_classes.fully_connected_layer import fully_connected_layer
from network_classes.network import network


net = network(784, 10)

net.add_layer(fully_connected_layer(784, 10))
net.add_layer(activation_layer(sigmoid,der_sigmoid))
net.add_layer(fully_connected_layer(10, 10))
net.add_layer(activation_layer(sigmoid,der_sigmoid))

