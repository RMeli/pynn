"""
Train neural network to learn the non-linear XOR function.
"""

from pynn.train import train
from pynn.nn import NeuralNetwork
from pynn.layer import Linear, Tanh

import numpy as np

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

nn = NeuralNetwork([
    Linear(input_size = 2, output_size = 2),
    Tanh(),
    Linear(input_size = 2, output_size = 2),
])

train(nn, inputs, targets)

for i, t in zip(inputs, targets):
    predicted = nn(i)
    print(i, np.around(predicted).astype(int), t)