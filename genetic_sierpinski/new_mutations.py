from utils import *

def mutate(chromosome: Chromosome, mutation_prob: float) -> Chromosome:
    # mutate a chromosome
    # by a chance
    for i in range(len(chromosome)):
        if np.random.random() < mutation_prob:
            chromosome[i] = random_affine()
    return chromosome