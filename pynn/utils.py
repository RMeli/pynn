from pynn.tensor import Tensor

import numpy as np

def size(*args: Tensor):
    if not len(args):
        raise ValueError
    
    n = args[0].size
    
    for t in args[1:]:
        if t.size != n:
            raise ValueError

    return n

def tanh(x : Tensor) -> Tensor:
    return np.tanh(x)

def tanh_derivative(x: Tensor) -> Tensor:
    return 1. - tanh(x)**2

def softplus(x: Tensor) -> Tensor:
    return np.log(1 + np.exp(x))

def sigmoid(x: Tensor) -> Tensor:
    return 1. / (1. + np.exp(-x))