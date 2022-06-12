import numpy as np
import random
from typing import Dict, List, Tuple
from utils import *
from new_mutations import mutate
from params import PARAMS

def fitness(chromosome: Chromosome, d : DistanceMetric = similarity_measure,
            sigma = 0.3, eta = 2,) -> float:
    """
    Calculates the fitness of a chromosome.
    eta = 2,3
    sigma = 0.1,0.3,0.5
    """
    # k = len(chromosome)
    # # xi is the contraction factor
    # xi = None # not sure how to calculate this
    # k_loss = np.exp(- k ** 2 / (4 * eta ** 2))
    # xi_loss = (1 - xi ** 10) * np.exp(- xi ** 2 / (4 * sigma ** 2))
    # return (d(sierpinski_128, apply_transformations(sierpinski_128, chromosome))
    #         * k_loss * xi_loss)
    return d(sierpinski_128, apply_transformations(sierpinski_128, chromosome))

def selection(item_score_map: List[Tuple[Chromosome, float]], num_Best: int, 
              )-> Tuple[List[Chromosome], List[Chromosome]]:
    # select the best members of the population
    # and the members that survived
    # return (best_fitness, bests, survived)
    bests = item_score_map[:num_Best]
    _, best_fitness = bests[0]
    def select(fitness): return fitness / best_fitness < np.random.random()
    survived = [chromo for chromo, fitness in item_score_map[num_Best:] if select(fitness)]
    bests = [chromo for (chromo, _) in bests]
    return bests, survived

# def crossover1(c1: Chromosome, c2: Chromosome) -> Tuple[Chromosome, Chromosome]:
#     # crossover two chromosomes
#     cPoint1 = c1.index(random.choice(c1))
#     cPoint2 = c2.index(random.choice(c2))
#     child1 = c1[0: cPoint1].extend(c2[cPoint2:])
#     child2 = c1[0: cPoint2].extend(c2[cPoint1:])
#     return child1, child2

def crossover1(c1: Chromosome, c2: Chromosome) -> Tuple[Chromosome, Chromosome]:
    np.random.shuffle(c1)
    np.random.shuffle(c2)
    c1[1], c2[1] = c2[1], c1[1]


def crossover_all(cs : List[Chromosome], crossProb : float) -> List[Chromosome]:
    # randomly pair up chromosomes
    # cross over them by a chance
    n = len(cs)
    np.random.shuffle(cs)
    for i in range(0, len(cs), 2):
        if np.random.random() < crossProb and i + 1 < n:
            crossover1(cs[i], cs[i+1])

def initialise_population(num_members: int, num_genes: int) -> Population:
    return [[random_affine() for _ in range(num_genes)] for _ in range(num_members)]

def evolve():
    best_fitness = 0
    population = initialise_population(100, 3)
    for i in range(PARAMS.NUM_OF_GENERATIONS):
        chromo_score_map = [(chromo, fitness(chromo)) for chromo in population]

        # descending order
        chromo_score_map = sorted(chromo_score_map, key=lambda x: x[1], reverse=True)
        _, best_fitness = chromo_score_map[0]
        bests, survived = selection(chromo_score_map, PARAMS.NUM_OF_BEST)
        crossover_all(survived, PARAMS.PROB_CROSS)
        mutate(survived, PARAMS.PROB_MUT)
        population = survived + bests
        if i % 10 == 0:
            print("Generation:", i, "Best fitness:", best_fitness)
        if best_fitness > PARAMS.TARGET_FITNESS:
            print("Generation:", i, "Best fitness:", best_fitness)
            break
    else:
        print("Generation:", i, "Best fitness:", best_fitness)


def test_selection():
    population = initialise_population(100, 3)
    chromo_score_map = [(chromo, fitness(chromo)) for chromo in population]
    chromo_score_map = sorted(chromo_score_map, key=lambda x: x[1], reverse=True)
    bests, survived = selection(chromo_score_map, PARAMS.NUM_OF_BEST)
    print(bests)
    print(survived)

def test_crossover():
    c1 = [1,1,1]
    c2 = [2,2,2]
    c3 = [3,3,3]
    c4 = [4,4,4]
    c5 = [5,5,5]
    # crossover1(c1, c2)
    print(c1)
    print(c2)
    population = [[np.ones((2, 3)) * i] * 3 for i in range(20)]
    crossover_all(population, 0.5)
    for i in population:
        print(i)


# test_selection()
# test_crossover()

evolve()