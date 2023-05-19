from python_code.helper_functions.activation_funcs import sigmoid, der_sigmoid


def test_sigmoid_given_any_when_zero_then_half():
    assert sigmoid(0) == 0.5


def test_der_sigmoid_given_any_when_zero_then_less_than_e():
    assert der_sigmoid(0) < 0.3
    assert der_sigmoid(0) > 0.2
