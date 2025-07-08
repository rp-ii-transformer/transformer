import numpy as np

def log_softmax(x):
    """
    log-softmax estável em NumPy
    """
    x_shift = x - np.max(x, axis=-1, keepdims=True)
    log_sum = np.log(np.sum(np.exp(x_shift), axis=-1, keepdims=True))
    return x_shift - log_sum

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def projection(x, W, b):
    return x @ W + b