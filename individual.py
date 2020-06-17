"""class for single individual/genome in the population"""
import numpy as np

from cppn import CPPN

class Individual:

    def __init__(self, ID):
        self.cppn = CPPN(novel=True)
        self.fitness = 0
        self.age = 0
        self.id = ID

    def mutate(self):
        self.cppn.mutate()

    def evaluate(self):
        self.cppn.evaluate()

    def print(self):
        self.cppn.print()


# testing
if __name__=='__main__':
    np.random.seed(1)
    ind = Individual(0)

    ind.print()

    # ind.evaluate()
    #
    # ind.mutate()
    #
    # ind.print()
    # ind.evaluate()