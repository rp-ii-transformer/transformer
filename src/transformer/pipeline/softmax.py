import numpy as np

def cross_entropy_loss(probs, target_idx):
    return -np.log(probs[target_idx] + 1e-9)  # adiciona epsilon para evitar log(0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def projection(x, W, b):
    return x @ W + b
