import numpy as np

from network_classes.layer import layer

class fully_connected_layer(layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size,1)
        self.bias = np.random.randn(output_size,1)

    def feed_forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def back_propagation(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T) # 10x1 * 1*
        input_gradient = np.dot(self.weights, output_gradient.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient