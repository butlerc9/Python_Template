import numpy as np

from python_code.network_classes.fully_connected_layer import fully_connected_layer

# test feed forward
# test back propagation
# test update weights
# test update bias


def test_fully_connected_layer_feed_forward_matrix_of_ones():
    layer = fully_connected_layer(2, 3)

    layer.weights = layer.weights * 0 + 1  # shift all weights to 1
    layer.bias = layer.bias * 0 + 1

    output = layer.feed_forward(np.ones([3, 2]))

    assert np.array_equal(output, np.array([[3, 3, 3], [3, 3, 3], [3, 3, 3]]))


def test_fully_connected_layer_feed_forward_matrix_of_ones():
    layer = fully_connected_layer(2, 3)

    layer.weights = layer.weights * 0 + 1  # shift all weights to 1
    layer.bias = layer.bias * 0 + 1

    layer.update_weights(0.1, np.array([1, 1, 1]))

    assert np.array_equal(layer.weights, np.array([0, 0, 0]))
