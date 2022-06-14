import numpy as np
import cv2
from genetic_sierpinski.new_utils import *

def affine_mat_to_mat_vec(affine_mat):
  mat = affine_mat[:, :2]
  vec = affine_mat[:, 2]
  return mat, vec

def mat_vec_to_affine(mat, vec):
  return np.concatenate((mat, vec.reshape((2, 1))), axis = 1)

def rotation(w):
  theta = np.random.uniform() * np.pi
  mat, vec = affine_mat_to_mat_vec(w)
  rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
  mat = np.dot(rotation_matrix, mat)
  new_map = mat_vec_to_affine(mat, vec)
  w[:, :] = new_map


def scale(w):
  # probably need to change this
  s = np.random.uniform(0.5, 1.5)
  (a, b, e,
   c, d, f) = w.reshape((6,))
  if np.random.random() < 0.5:
    w[0, 0] = a * s
  else:
    w[0, 1] = b * s

def skew(w):
  s = np.random.uniform(0.0, 1.0)
  (a, b, e,
   c, d, f) = w.reshape((6,))
  if np.random.random() < 0.5:
    w[1, 0] = a * s + b
    w[1, 1] = c * s + d
  else:
    w[0, 0] = a + b * s
    w[0, 1] = c + d * s

def translation( w):
  (a, b, e,
   c, d, f) = w.reshape((6,))
  r = (e + f) / 4
  x = np.random.uniform(-r, r)
  w = w.copy()
  if np.random.random() < 0.5:
    w[0, 2] = e + x
  else:
    w[1, 2] = f + x


def mutate(cs : List[Chromosome], mutProb : float):
    for chromo in cs:
        if mutProb > np.random.random():
            if np.random.random() < 0.5: map_mutation(chromo)
            else:                        map_perturbation(chromo)
            # if ifsMutProb > np.random.random(): # is ifs mutation
            #     if np.random.random() < 0.5: chromo.remove(np.random.choice(chromo))
            #     else:                        chromo.append(random_affine())
            # else: # is map mutation
            #     if np.random.random() < 0.5: map_mutation(chromo)
            #     else:                        map_perturbation(chromo)

# both types of map mutation need to repair if out of bounds

def map_mutation(chromosome: Chromosome):
    """
    mutate a chromosome by changing a random affine map
    """
    flag = np.random.random()
    i = np.random.randint(0, len(chromosome.genes))
    item = chromosome.genes[i]
    if flag < 0.25:   rotation(item)
    elif flag < 0.5:  translation(item)
    elif flag < 0.75: skew(item)
    else:             scale(item)


def map_perturbation(chromosome: Chromosome):
    i = np.random.randint(0, len(chromosome.genes))
    map = chromosome.genes[i]
    flag = np.random.random()
    if flag < 1 / 6:   map[0,0] += np.random.random() - 0.5
    elif flag < 2 / 6: map[0,1] += np.random.random() - 0.5
    elif flag < 3 / 6: map[1,0] += np.random.random() - 0.5
    elif flag < 4 / 6: map[1,1] += np.random.random() - 0.5
    elif flag < 5 / 6: map[0,2] += np.random.random() - 0.5
    else:              map[1,2] += np.random.random() - 0.5
