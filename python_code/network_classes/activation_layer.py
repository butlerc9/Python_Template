import numpy as np

class activation_layer():
    def __init__(self, activation_function, activation_function_derivative):
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        
    def feed_forward(self, input_array: np.array) -> np.array:
        output_vector = self.activation_function(input_array)
        return output_vector
    
    def backpropagation(self, input_gradient: np.array) -> np.array:
        output_gradient = self.activation_function_derivative(input_gradient)
        return output_gradient