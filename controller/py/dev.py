import numpy as np
from ..NNFromScratch import activation

x = np.random.randn(6)
print("Original: ",x)

sigmoid_x = activation.sigmoid(x)
print("Sigmoid: ",sigmoid_x)

softmax_x = activation.softmax(x)
print("Softmax: ",softmax_x)

relu_x = activation.ReLU(x)
print("ReLU: ",relu_x)

#To run this: python -m controller.py.dev