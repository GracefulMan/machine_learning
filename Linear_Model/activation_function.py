import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
