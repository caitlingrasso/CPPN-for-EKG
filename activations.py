import numpy as np

# TODO: implement activation functions (sigmoid, sin, cos, tanh, relu)

def sigmoid(x):  # output (0,1)... rescaled sigmoid? What is the time range/scale?
    s = 1 / (1 + np.exp(-x))
    return s

def sin():
    pass

def cos():
    pass

def relu():
    pass

def tanh():
    pass