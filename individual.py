import numpy as np

from cppn import CPPN

class Individual:

    def __init__(self):
        self.cppn = CPPN(fixed=True)

    def mutate(self):
        self.cppn.mutate()

    def evaluate(self):
        self.cppn.evaluate()

    def print(self):
        self.cppn.print()

if __name__=='__main__':
    # np.random.seed(1)
    ind = Individual()
    # ind.print()
    ind.evaluate()
    # ind.mutate()