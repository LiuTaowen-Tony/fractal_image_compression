from GeneticAlgorithm import *

chromos = [Chromosome([random_affine() for _ in range(3)]) for _ in range(50)]
pop = Population(chromos, 5, 60, 50, 0.3, 0.9, 0.5, 0.6, 30)
for i in range(10000):
    pop.selection()
    pop.crossover()
    pop.mutate()
    pop.repair()

    if i % 1000 == 0:
        genes = pop.best[0][0].genes
        for mat in genes:
            print(mat)
        print(pop.best[0][1])

print("the best fitness value was " + str(pop.best[0][1]))
pic = np.ones((128,128), dtype = np.uint8)*255
genes = pop.best[0][0].genes
for mat in genes:
    print(mat)
for _ in range(5):
     pic = w(pic, genes)
     cv2.imshow("result", pic)
     cv2.waitKey(0)