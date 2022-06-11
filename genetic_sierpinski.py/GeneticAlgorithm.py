# flake8: noqa E501
import imp
import numpy as np
import cv2
import random
from typing import List, Function
from utils import sierpinski_128, apply_transformations, similarity_measure

# Chromosomes are a list of affine matrices
Chromosome = List[np.ndarray]
DistanceMetric = Function[(np.ndarray, np.ndarray), float]



def fitness(chromosome: Chromosome, d : DistanceMetric,
            sigma = 0.3, eta = 2,) -> float:
    """
    Calculates the fitness of a chromosome.
    eta = 2,3
    sigma = 0.1,0.3,0.5
    """
    k = len(chromosome)
    # xi is the contraction factor
    xi = None # not sure how to calculate this
    k_loss = np.exp(- k ** 2 / (4 * eta ** 2))
    xi_loss = (1 - xi ** 10) * np.exp(- xi ** 2 / (4 * sigma ** 2))
    return (d(sierpinski_128, apply_transformations(sierpinski_128, chromosome))
            * k_loss * xi_loss)

def random_affine() -> np.ndarray:
    e, f = np.random.uniform(0., 1., 2)
    a, b = np.random.uniform(-e, 1-e, 2)
    if (a + b + e) > 1 or (a + b - e) < 0:
        a, b = a / 2, b / 2
    c, d = np.random.uniform(-f, 1-f, 2)
    if (c + d + f) > 1 or (c + d - f) < 0:
        c, d = c / 2, d / 2
    return np.array([[a, b, e],
                     [c, d, f]])



class Population():
    def __init__(self, chromosomes, bestMax, chromLenMax, popMax, crossProb, mutProb, ifsMutProb):
        self.members = chromosomes
        self.bestMax = bestMax
        self.chromLenMax = chromLenMax
        self.popMax = popMax
        self.crossProb = crossProb
        self.mutProb = mutProb
        self.ifsMutProb = ifsMutProb



    def selection(self, gap):
        best = {} # may need to add initial value of 0 or something so can be overwritten
        # we want to maximise the fitness no?
        chromo_fit = {
            member: fitness(member, similarity_measure) for member in self.members
        }
        sorted_chromosomes = sorted(chromo_fit.items(), key=lambda x: x[1], reverse=True)
        best = sorted_chromosomes[0:gap][0]
        # idk what to do next

    def crossover(self):
        # after crossover, do we kill the parents?
        # we don't kill in this implementation
        np.random.shuffle(self.members)
        pairs = (i, j for i , j in zip(self.members[::2], self.members[1::2]))
        for i, j in pairs:
            if self.crossProb > random.random():
                cPoint1 = i.index(random.choice(i))
                cPoint2 = j.index(random.choice(j))
                child1 = i[0: cPoint1].extend(j[cPoint2:])
                child2 = i[0: cPoint2].extend(j[cPoint1:])
                self.members.extend([child1, child2])

    def mutate(self):
        for chromo in self.members:
            if self.mutProb > np.random.random():

                if self.ifsMutProb > np.random.random():
                    # is ifs mutation
                    # either add or remove a gene
                    if np.random.random() < 0.5: chromo.remove(np.random.choice(chromo))
                    else:                        chromo.append(random_affine())
                else: # is map mutation
                    if np.random.random() < 0.5:
                        map_mutation(chromo)
                    else:
                        map_perturation(chromo)
                

def evolve():
    t = 0
    chromos = []
    population = Population(chromos, bestMax, chromLenMax, popMax, crossProb, mutProb, ifsMutProb)
    max_fitness = 0
    while(max_fitness < TARGET_FITNESS or t > MAX_GENERATION):
        population.selection(gap)
        population.crossover()
        population.mutate()
        population.repair()
        t += 1
        max_fitness = 0





# Todos :
# Random Affine
# fitness function (we have similarity and we need panelise the other two params)
# [0-1]^2 -> [0-1] # fixed
# combine mutation