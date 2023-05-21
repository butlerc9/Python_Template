import numpy as np


class fully_connected_layer:
    def __init__(self, input_size: int, layer_size: int):
        # randomly start weights and biases between 0 and 1
        self.weights = np.random.rand(input_size, layer_size)
        self.bias = np.random.rand(layer_size)

    def feed_forward(self, input_array: np.array) -> np.array:
        self.activation = np.matmul(input_array, self.weights) + self.bias
        return self.activation

    def back_propagation(self, input_gradient: np.array) -> np.array:
        self.gradient = self.weights * input_gradient
        return self.gradient

    def update_weights(self, learning_rate: float, input_gradient: np.array) -> None:
        number_of_observations_part_of_batch = np.shape(input_gradient)[0]
        average_gradient = (
            sum(np.dot(self.weights, input_gradient))
            / number_of_observations_part_of_batch
        )
        self.weights -= average_gradient * learning_rate

    def update_bias(self, learning_rate: float, input_gradient: np.array) -> None:
        self.bias -= learning_rate * input_gradient
