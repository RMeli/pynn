"""
Train neuronal network to learn handwritten digits from the MNIST database.

The MNIST database is composed of 8x8 images of handwritten digits.
"""

from matplotlib import pylab as plt
from sklearn import datasets

# Load MNIST database
images, targets = datasets.load_digits(return_X_y=True)

plt.imshow(images[0].reshape(8, 8), cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
