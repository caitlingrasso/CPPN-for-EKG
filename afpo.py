"""Age-Fitness Pareto Optimization"""

import numpy as np
from copy import deepcopy
import operator
import argparse
import sys

from genome import Genome

class AFPO:

    def __init__(self, gens, popsize):

        self.gens = gens
        self.target_size = popsize
        self.next_available_id = 0

        self.population = self.create_initial_population()

        self.fits_per_gen = np.zeros(self.gens)

    def run(self):
        self.perform_first_generation()

        for g in range(1, self.gens):
            self.perform_one_generation(g)
            self.fits_per_gen[g] = self.find_best().fitness

        best = self.find_best()
        return best, self.fits_per_gen

    def create_initial_population(self):
        population = {}
        for i in range(self.target_size):
            population[i] = Genome(self.next_available_id)
            self.next_available_id += 1
        return population

    def perform_first_generation(self):
        for i in self.population:
            self.population[i].evaluate()
        self.print_best(0)

    def perform_one_generation(self, g):
        self.increase_age()
        children = self.generate_children()
        children = self.inject(children)
        children = self.evaluate(children)
        self.extend(children)
        self.reduce()
        self.print_best(g)

    def print(self, g):
        print(g, end=' ')
        for i in self.population:
            self.population[i].print()
        print()

    def print_best(self, g):
        print(g, end=' ')
        best = self.find_best()
        best.print()
        print()

    def increase_age(self):
        for i in self.population:
            self.population[i].age += 1

    def generate_children(self):
        children = []
        for i in range(len(self.population)):
            parent = self.tournament_selection()
            child = deepcopy(self.population[parent])
            child.id = self.next_available_id
            self.next_available_id += 1
            child.mutate()
            children.append(child)
        return children

    def tournament_selection(self):
        p1 = np.random.randint(self.target_size)
        p2 = np.random.randint(self.target_size)
        while p1 == p2:
            p2 = np.random.randint(self.target_size)
        if self.population[p1].fitness < self.population[p2].fitness:
            return p1
        else:
            return p2

    def evaluate(self, population):
        for i in range(len(population)):
            population[i].evaluate()
        return population

    def inject(self, children):
        """Adds random individual to population"""
        children.append(Genome(self.next_available_id, novel=True))
        self.next_available_id += 1
        return children

    def extend(self, children):
        for i, j in enumerate(range(len(self.population), 2*len(self.population)+1)):
            self.population[j] = children[i]

    def reduce(self):
        """Reduces population by removing dominated individuals"""
        pf = self.find_pareto_front()
        pareto_size = len(pf)

        if pareto_size > self.target_size:
            self.target_size = pareto_size

        while len(self.population) > self.target_size:

            pop_size = len(self.population)

            ind1 = np.random.randint(pop_size)
            ind2 = np.random.randint(pop_size)
            while ind1 == ind2:
                ind2 = np.random.randint(pop_size)

            if self.dominates(ind1, ind2): # ind1 dominates

                for i in range(ind2, len(self.population) - 1):
                    self.population[i] = self.population.pop(i+1)

            elif self.dominates(ind2, ind1): # ind2 dominates

                for i in range(ind1, len(self.population) - 1):
                    self.population[i] = self.population.pop(i+1)

    def dominates(self, ind1, ind2):
        return self.population[ind1].dominates_other(self.population[ind2])

    def find_best(self):
        """Returns individual in population with the highest fitness"""
        sorted_pop = sorted(self.population.values(), key=operator.attrgetter('fitness'), reverse=False)
        return sorted_pop[0]

    def find_pareto_front(self):
        """Returns indices of non-dominated individuals in the population"""

        pareto_front = []

        for i in self.population:
            i_is_dominated = False
            for j in self.population:
                if i!=j:
                    if self.dominates(j, i):
                        i_is_dominated = True
            if not i_is_dominated:
                pareto_front.append(i)

        return pareto_front

    def find_pareto_front_individuals(self):
        pf = self.find_pareto_front()
        pf_inds = []
        for i in pf:
            pf_inds.append(self.population[i])
        return pf_inds

if __name__=='__main__':

    def parse_args(args):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gens', default=10, type=int)
        parser.add_argument('--popsize', default=5, type=int)
        return parser.parse_args(args)

    args = sys.argv[1:]
    args = parse_args(args)

    afpo = AFPO(gens=args.gens, popsize=args.popsize)
    best, fits_per_gen = afpo.run()

    print(best.fitness)
    best.CPPN_summary()
