import numpy as np

def softmax(x):
    x_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x_exp / np.sum(x_exp, axis=-1, keepdims=True)
