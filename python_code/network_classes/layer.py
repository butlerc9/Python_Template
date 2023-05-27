import numpy as np

# create base class layer
# nothing will be inherited, just to make sure correct error is raised if feature not implemented

class layer():
    def __init__(self):
        self.input = None
        self.output = None
    
    def feed_forward(self, input_array: np.array) -> np.array:
        raise NotImplementedError()
    
    def back_propagation(self, output_gradient: np.array, learning_rate: float) -> np.array:
        raise NotImplementedError()