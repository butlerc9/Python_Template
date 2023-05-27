import numpy as np

from network_classes.layer import layer

class activation_layer(layer):
    def __init__(self, activation_function, activation_function_derivative):
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def feed_forward(self, input_array: np.array) -> np.array:
        self.input = input_array
        
        self.output = self.activation_function(input_array)

        return self.output
    
    def back_propagation(self, output_gradient: np.array, learning_rate: float) -> float:
        # input_grad = output_grad * f'(input)
        input_gradient = np.multiply(output_gradient , self.activation_function_derivative(self.input))
        
        return input_gradient