import numpy as np
from network_classes.activation_layer import activation_layer
from network_classes.fully_connected_layer import fully_connected_layer


class network:
    def __init__(self, input_size: int, output_size: int):
        self.layers = []
        self.input_size = input_size
        self.output_size = output_size

    def add_layer(self, layer: fully_connected_layer | activation_layer):
        self.layers.append(layer)

    def cost_function(self, estimation_vector, actual_vector):
        return ((estimation_vector - actual_vector) ** 2).sum()

    def predict(self, input_array: np.array):
        activation_vector = input_array
        for layer in self.layers:
            activation_vector = layer.feed_forward(activation_vector)

        return activation_vector
