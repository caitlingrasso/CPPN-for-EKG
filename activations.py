import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def relu(x):
    return np.maximum(np.zeros(x.shape),x)

def tanh(x):
    return np.tanh(x)

def abs(x):
    return np.abs(x)

def identity(x):
    return x

def normalize(x):
    x -= np.min(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = x/np.max(x)
    x = np.nan_to_num(x)
    x *= 2
    x -= 1
    return x

def rescaled_positive_sigmoid(x, x_min=0, x_max=500):
    return (x_max - x_min) * sigmoid(x) + x_min