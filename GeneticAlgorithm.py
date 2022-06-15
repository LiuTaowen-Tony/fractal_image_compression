# flake8: noqa E501
import random
import re
import numpy as np
from copy import deepcopy
from genetic_sierpinski.new_utils import *
from genetic_sierpinski.mutations import *


class Chromosome():
    def __init__(self, affines):
        self.genes = affines
    def __getitem__(self, i): return self.genes[i]
    def __len__(self, i): return len(self.genes)


class Population():
    def __init__(self, chromosomes, bestMax, chromLenMax, popMax, crossProb, mutProb, ifsMutProb, STDCT, STDCP):
        self.members = chromosomes
        self.bestMax = bestMax
        self.chromLenMax = chromLenMax
        self.popMax = popMax
        self.crossProb = crossProb
        self.mutProb = mutProb
        self.ifsMutProb = ifsMutProb
        self.best = [Chromosome([])]
        self.STDCT = STDCT
        self.STDCP = STDCP

    def selection(self):
        memberFitBestSorted =  sorted([(chromo, fitness(chromo, self.STDCT, self.STDCP)) for chromo in self.members], key=lambda x: x[1], reverse=True)
        self.best = memberFitBestSorted[:self.bestMax]

    def crossover(self):
        newMembers = [i[0] for i in self.best]
        
        for i in self.best:
            parent1 = i[0]
            parent2 = random.choice(self.best)[0]
            cGene1 = random.choice(parent1.genes)
            cGene2 = random.choice(parent2.genes)
            cPoint1 = npGetArrInd(parent1.genes, cGene1)
            cPoint2 = npGetArrInd(parent2.genes, cGene2)
            child1 = deepcopy(parent1.genes[0: cPoint1])
            child1.extend(deepcopy(parent2.genes[cPoint2:]))
            child2 = deepcopy(parent2.genes[0: cPoint2])
            child2.extend(deepcopy(parent1.genes[cPoint1:]))
            newMembers.extend([Chromosome(child1), Chromosome(child2)])
        self.members = newMembers[:self.popMax]
        
        # i = 0
        # while i < len(self.members):
        #     if self.crossProb > np.random.random():
        #         parent1 = self.members[i]
        #         self.members.remove(parent1)
        #         try:
        #             parent2 = np.random.choice(self.members)
        #         except:
        #             newMembers.append(parent1)
        #             break
        #         self.members.remove(parent2)
        #         cGene1 = random.choice(parent1.genes)
        #         cGene2 = random.choice(parent2.genes)
        #         cPoint1 = npGetArrInd(parent1.genes, cGene1)
        #         cPoint2 = npGetArrInd(parent2.genes, cGene2)
        #         child1 = parent1.genes[0: cPoint1]
        #         child1.extend(parent2.genes[cPoint2:])
        #         child2 = parent2.genes[0: cPoint2]
        #         child2.extend(parent1.genes[cPoint1:])
        #         newMembers.extend([Chromosome(child1), Chromosome(child2)])
        #         i = 0
        #     else: i += 1
        # self.members.extend(newMembers)

    def mutate(self):
        i = 0
        while i < len(self.members):
            if self.mutProb > np.random.random():
                if self.ifsMutProb > np.random.random():
                    if np.random.random() < 0.5 and len(self.members[i].genes) != 1:
                        self.members[i].genes.pop(npGetArrInd(self.members[i].genes, random.choice(self.members[i].genes)))
                        val = 1
                    else:
                        self.members[i].genes.append(random_affine())
                        val = 2

                else:
                    if np.random.random() < 0.5:
                        map_mutation(self.members[i])
                        val = 3
                    else:
                        map_perturbation(self.members[i], self.STDCT, self.STDCP)
                        val = 4
            i += 1

    def repair(self):
        #controls growth: stops chromosomes from having too many genes and the population being too large
        bestChromos = [i[0] for i in self.best]
        bestChromos.extend(self.members)
        self.members = bestChromos
        self.members = self.members[:self.popMax]
        self.members = [Chromosome(i.genes[:self.chromLenMax]) for i in self.members]