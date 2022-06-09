# flake8: noqa E501
import random


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
        best = {} # may need to add initial value of 0 or something so can be overwritten
        for i in self.members:
            minBestIndex = best.values().index(min(best.values()))
            if fitness(i) > best.values()[minBestIndex]:
                best.update({i, fitness(i)})
                if len(best) > self.bestMax:
                    best.pop(best.keys()[minBestIndex])
        self.members = best.keys()

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
                        newMembers.append(randAffine())
                else:
                    # do (maybe one of many) map mutations
                i = 0
        self.members.extend(newMembers)