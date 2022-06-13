import numpy as np
import random
from typing import Dict, List, Tuple
from new_utils import *
from new_mutations import mutate
import params as PARAMS
from numba import njit
from copy import deepcopy

def selection(item_score_map: List[Tuple[Chromosome, float]], num_Best: int, 
              )-> Tuple[List[Chromosome], List[Chromosome]]:
    # select the best members of the population
    # and the members that survived
    # return (bests, survived)
    bests = item_score_map[:num_Best]
    _, best_fitness = bests[0]
    def select(fitness): return (1 - 10 / np.log(fitness)) < np.random.random()
    survived = [chromo for chromo, fitness in item_score_map[num_Best:] if select(fitness)]
    bests = [chromo for (chromo, _) in bests]
    return bests, survived

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
        chromo_score_map = sorted(chromo_score_map, key=lambda x: x[1])
        _, best_fitness = chromo_score_map[0]
        bests, survived = selection(chromo_score_map, PARAMS.NUM_OF_BEST)
        num_survided = len(survived)
        bests_copy = deepcopy(bests[::3])
        survived.extend(bests_copy)
        crossover_all(survived, PARAMS.PROB_CROSS)
        for chromo in survived:
            for gene in chromo:
                gene += np.random.normal(0, 0.1, gene.shape)
                gene[gene > 1] = 1
                gene[:] = np.abs(gene)
        population = survived + bests

        while len(population) < PARAMS.POPULATION_SIZE:
            index = np.random.randint(0, len(population))
            population.append(deepcopy(population[index]))

        # for chromo in population:
        #     for gene in chromo:
        #         gene[gene > 1] = 1
        #         gene[:] = np.abs(gene)
        if i % 1000 == 0:
            print("Generation:", i, "Best fitness:", best_fitness, "num_survided:", num_survided)
            print(bests[0])
            for _, score in chromo_score_map[::5]:
                print(score)
        # if best_fitness < PARAMS.TARGET_FITNESS:
        #     print("Generation:", i, "Best fitness:", best_fitness)
        #     break
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