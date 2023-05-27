import numpy as np

from network_classes.layer import layer

class network():
    def __init__(self):
        self.layers = []
        self.total_loss = 0.0
        self.correct_predictions_during_current_epoch = 0

    def add_layer(self, layer: layer):
        self.layers.append(layer)

    def calculate_cost_vector(self, x: int, y: int) -> np.array:
        self.total_loss += ((x - y.T) ** 2).sum() # This is the average for one training example, we will take the average of this over the epoch
        cost_vector = 2 * (x - y.T)
        # return output gradient      
        return cost_vector
    
    def feed_forward_single_example(self, x: int):
        
        layer_activation = x
        
        for layer in self.layers:
            layer_activation = layer.feed_forward(layer_activation)      
        

        return layer_activation

    def back_propagation_single_example(self, x: int, learning_rate: float):
        
        output_gradient = x
        
        for layer in reversed(self.layers):

            output_gradient = layer.back_propagation(output_gradient,learning_rate)

        return output_gradient

    
    def train(self, x_train: np.array, y_train: np.array, learning_rate: float) -> np.array:
        
        for i in range(100): # run 10 epochs
            for x, y in zip(x_train,y_train):
                final_layer_output = self.feed_forward_single_example(x)
                network_output_derivative = self.calculate_cost_vector(final_layer_output, y)
                self.back_propagation_single_example(network_output_derivative, learning_rate)
            
            print(self.total_loss/y_train.shape[0])
            self.total_loss = 0

            
                