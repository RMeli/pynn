from pynn.utils import size

import numpy as np
import pytest

def test_size_no_args():
    with pytest.raises(ValueError):
        n = size()

def test_size_error():
    x = np.ones(10)
    y = np.ones(10)
    z = np.ones(5)


    with pytest.raises(ValueError):
        n = size(x, y, z)

def test_size_one(n: int = 10):
    x = np.ones(n)

    assert size(x) == n

def test_size(n: int = 10):
    a = np.ones(n)
    b = np.ones(n)
    c = np.ones(n)
    d = np.ones(n)

    assert size(a, b, c, d) == n

