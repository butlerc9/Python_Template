import numpy as np

from python_code.network_classes.layer import layer

class activation_layer(layer):
    def __init__(self, activation_function, activation_function_derivative):
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def feed_forward(self, input_array: np.array) -> np.array:
        self.input = input_array
        
        self.output = 