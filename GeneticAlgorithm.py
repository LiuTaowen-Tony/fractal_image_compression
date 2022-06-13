# flake8: noqa E501
import random
import numpy as np
from genetic_sierpinski.new_utils import *
from genetic_sierpinski.mutations import *


class Chromosome():
    def __init__(self, affines):
        self.genes = affines


class Population():
    def __init__(self, chromosomes, bestMax, chromLenMax, popMax, crossProb, mutProb, ifsMutProb):
        self.members = chromosomes
        self.bestMax = bestMax
        self.chromLenMax = chromLenMax
        self.popMax = popMax
        self.crossProb = crossProb
        self.mutProb = mutProb
        self.ifsMutProb = ifsMutProb

    def selection(self):
        memberFitBestSorted =  sorted([(chromo, fitness(chromo)) for chromo in self.members], key=lambda x: x[1])
        best = memberFitBestSorted[:self.bestMax]
        def select(fitness): return (1 - 10 / np.log(fitness)) < np.random.random()    
        survived = [chromo for chromo, fitness in  memberFitBestSorted[self.bestMax:] if select(fitness)]
        survived.extend([chromo for (chromo, _) in best])
        self.members = survived

    def crossover(self):
        newMembers = []
        i = 0
        while i < len(self.members):
            if self.crossProb > random.random():
                parent1 = self.members[i]
                self.members.remove(parent1)
                try:
                    parent2 = random.choice(self.members)
                except:
                    break
                self.members.remove(parent2)
                cPoint1 = parent1.genes.index(random.choice(parent1.genes))
                cPoint2 = parent2.genes.index(random.choice(parent2.genes))
                child1 = parent1.genes[0: cPoint1].extend(parent2.genes[cPoint2:])
                child2 = parent2.genes[0: cPoint2].extend(parent1.genes[cPoint1:])
                newMembers.extend([child1, child2])
                i = 0
        self.members.extend(newMembers)

    def mutate(self):
        newMembers = []
        i = 0
        while i < len(self.members):
            if self.mutProb > random.random():
                if self.ifsMutProb > random.random():
                    if random.random() < 0.5:
                        self.members[i].gene.remove(random.choice(self.members[i].gene))
                    else:
                        newMembers.append(random_affine())
                else:
                    if np.random.random() < 0.5: map_mutation(self.members[i])
                    else:                        map_perturbation(self.members[i])

                i = 0
        self.members.extend(newMembers)

    def repair(self):
        #controls growth: stops chromosomes from having too many genes and the population being too large
        self.members = self.members[:self.popMax]
        self.members = [i.genes[:self.chromLenMax] for i in self.members]