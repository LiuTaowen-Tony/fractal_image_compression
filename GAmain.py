from GeneticAlgorithm import *

chromos = [Chromosome([random_affine() for _ in range(3)]) for _ in range(50)]
pop = Population(chromos, 5, 6, 50, 0.1, 0.01, 0.5)
for i in range(1000):
    pop.selection()
    pop.crossover()
    pop.mutate()
    pop.repair()
