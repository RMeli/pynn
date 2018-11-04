from pynn.layer import Layer

import numpy as np
import pytest

def test_layer():

    L = Layer()

    with pytest.raises(NotImplementedError):
        L.forward(np.ones(10))

    with pytest.raises(NotImplementedError):
        L.backward(np.ones(10))
