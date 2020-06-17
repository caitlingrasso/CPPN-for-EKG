import numpy as np

from cppn import CPPN

class Individual:

    def __init__(self):
        self.cppn = CPPN(novel=True)
        self.fitness = 0
        self.age = 0

    def mutate(self):
        self.cppn.mutate()

    def evaluate(self):
        self.cppn.evaluate()

    def print(self):
        self.cppn.print()

if __name__=='__main__':
    # np.random.seed(3)
    ind = Individual()

    ind.print()
    ind.evaluate()

    ind.mutate()

    ind.print()
    ind.evaluate()