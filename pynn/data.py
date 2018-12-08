"""
Tools for data manipulation.

Inputs are feed into the neural network in batches.
"""

from pynn.tensor import Tensor

from abc import ABC, abstractmethod
from typing import Iterator, NamedTuple

import numpy as np

# NamedTupled defining a batch of inputs and targets for training
Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])


class DataIterator(ABC):
    """
    Data iterator.
    """

    @abstractmethod
    def __call__(self, inputs: Tensor, target: Tensor) -> Iterator[Batch]:
        pass


class BatchIterator(DataIterator):
    """
    Iterate over data in batches (with possible re-shuffling of the batches).
    """

    def __init__(self, batch_size: int = 32, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        # Shuffle batches but not within batches
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size

            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]

            yield Batch(batch_inputs, batch_targets)
