import numpy as np

from python_code.network_classes.activation_layer import activation_layer
from python_code.helper_functions.activation_funcs import sigmoid, der_sigmoid


def test_activation_layer_sigmoid_feed_forward_given_array_of_zeros_can_apply_activation_function_element_wise():
    layer = activation_layer(sigmoid, der_sigmoid)

    number_of_neurons = 3
    input_size = 2
    # create test input
    input_array = np.zeros([number_of_neurons, input_size])
    # feed test input through array
    output = layer.feed_forward(input_array)
    # create similar matrix of 0.5 because sigmoid(0) = 0.5
    actual = np.zeros([number_of_neurons, input_size]) + 0.5
    # this will create a matrix containing all true of false values
    comparison = actual == output
    # make sure all elements match
    assert comparison.all()


def test_activation_layer_sigmoid_back_propagation_given_array_of_zeros_can_apply_activation_function_element_wise():
    layer = activation_layer(sigmoid, der_sigmoid)

    number_of_neurons = 3
    input_size = 2
    # create test input
    input_array = np.zeros([number_of_neurons, input_size])
    # feed test input through array
    output = layer.back_propagation(input_array)
    # this will create a matrix containing all true of false values
    lower_bound_check = output < 0.3
    upper_bound_check = output > 0.2
    # make sure all elements are true
    assert lower_bound_check.all()
    assert upper_bound_check.all()
