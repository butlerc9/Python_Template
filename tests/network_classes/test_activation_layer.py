import numpy as np

from python_code.network_classes.activation_layer import activation_layer
from python_code.helper_functions.activation_funcs import sigmoid, der_sigmoid

def test_feed_forward_element_wise_activation():
    layer = activation_layer(sigmoid, der_sigmoid)

    input_array = np.zeros([2,3])

    output = layer.feed_forward(input_array)

    assert np.array_equal(np.full([2,3], 0.5),output)
    assert np.array_equal(np.full([2,3], 0.5),layer.output)