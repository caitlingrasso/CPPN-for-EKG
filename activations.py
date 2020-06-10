import numpy as np

# TODO: implement activation functions (sigmoid, sin, cos, tanh, relu)

def sigmoid(x):  # output (0,1)... rescaled sigmoid? What is the time range/scale?
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

def identity(x):
    return x