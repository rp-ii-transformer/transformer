from .common import xp

def log_softmax(x):
    """
    log-softmax estável em NumPy
    """
    x_shift = x - xp.max(x, axis=-1, keepdims=True)
    log_sum = xp.log(xp.sum(xp.exp(x_shift), axis=-1, keepdims=True))
    return x_shift - log_sum

def softmax(x):
    exp_x = xp.exp(x - xp.max(x, axis=-1, keepdims=True))
    return exp_x / xp.sum(exp_x, axis=-1, keepdims=True)

def projection(x, W, b):
    return x @ W + b
