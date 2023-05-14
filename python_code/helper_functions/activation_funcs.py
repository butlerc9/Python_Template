import math


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.e**-x)


def der_sigmoid(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))
