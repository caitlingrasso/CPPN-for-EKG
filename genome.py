"""class for single individual/genome in the population"""
import numpy as np

from cppn import CPPN

class Genome:

    def __init__(self, ID, novel=False):
        self.cppn = CPPN(novel=novel)
        self.fitness = 0
        self.age = 0
        self.id = ID

    def mutate(self):
        self.cppn.mutate()

    def evaluate(self):
        # self.fitness = self.cppn.evaluate()
        self.fitness = np.random.randint(20)

    def CPPN_summary(self):
        self.cppn.print()

    def print(self):
        print('[ id:', self.id, 'fitness:', self.fitness, ']', end='')

    def dominates_other(self, other):

        if self.age == other.age and self.fitness == other.fitness:
            return self.id > other.id
        elif self.age <= other.age and self.fitness >= other.fitness:
            return True
        else:
            return False


