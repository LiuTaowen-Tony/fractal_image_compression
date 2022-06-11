import numpy as np
import random
from typing import Dict, List, Function, Tuple
from utils import *
from mutations import mutate

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


def selection(cs: Population, num_Best: int, 
        d = similarity_measure)-> Tuple[float, Population, Population]:
    # select the best members of the population
    # and the members that survived
    # return (best_fitness, bests, survived)
    item_score_map = [ (chromo, fitness(chromo, d)) for chromo in cs ]
    # descending order
    item_score_map = sorted(item_score_map, key=lambda x: x[1], reverse=True)
    bests = item_score_map[:num_Best]
    survived = []
    _, best_fitness = bests[0]
    for (chromo, fitness) in survived:
        if np.random.random() < best_fitness / fitness:
            survived.append(chromo)
    bests = [chromo for (chromo, fitness) in bests]
    return best_fitness, bests, survived

def crossover1(c1: Chromosome, c2: Chromosome) -> Tuple[Chromosome, Chromosome]:
    # crossover two chromosomes
    cPoint1 = c1.index(random.choice(c1))
    cPoint2 = c2.index(random.choice(c2))
    child1 = c1[0: cPoint1].extend(c2[cPoint2:])
    child2 = c1[0: cPoint2].extend(c2[cPoint1:])
    return child1, child2

def crossover_all(cs : List[Chromosome], crossProb : float) -> List[Chromosome]:
    # randomly pair up chromosomes
    # cross over them by a chance
    np.random.shuffle(cs)
    pairs = (i, j for i , j in zip(cs[::2], cs[1::2]))
    result = []
    for i, j in pairs:
        if crossProb > random.random(): result.extend(crossover1(i, j))
        else:                           result.extend((i, j))
    return result

def initialise_population(num_members: int, num_genes: int) -> Population:
    return [ [random_affine() for _ in range(num_genes)] for _ in range(num_members) ]


def evolve():
    t = 0
    max_fitness = 0
    population = initialise_population(100, 3)
    while(max_fitness < TARGET_FITNESS or t > MAX_GENERATION):
        best_fitness, bests, survived = selection(population, 10)
        _, max_fitness = bests[0]
        survived = crossover_all(survived, 0.5)
        mutate(survived, 0.5, 0.5)
        population = survived + bests
        t += 1

