from pynn.layer import Layer, Linear

import numpy as np
import pytest

def test_layer():

    L = Layer()

    with pytest.raises(NotImplementedError):
        L.forward(np.ones(10))

    with pytest.raises(NotImplementedError):
        L.backward(np.ones(10))

def test_linear_forward_1d():

    L = Linear(1, 1)

    assert L.params["w"].size == 1
    assert L.params["b"].size == 1

    L.params["w"] = 3
    L.params["b"] = -1

    assert L.forward(2) == 5

def test_linear_forward():

    L = Linear(3, 2)

    assert L.params["w"].size == 6
    assert L.params["w"].shape[0] == 3
    assert L.params["w"].shape[1] == 2
    assert L.params["b"].size == 2

    L.params["w"] = np.array([[3, 3], [2, 2], [1, 1]])
    L.params["b"] = np.array([-1, -2])

    inputs = np.ones(3)
    outputs = np.array([5, 4])

    assert np.allclose(L.forward(inputs), outputs)

def test_linear_backward_1d():
    pass

def test_linear_backward():
    pass
