from xml.dom import NotFoundErr
import numpy as np
import cv2
from numba import njit, jit

from typing import List, Tuple, Callable

Chromosome = List[np.ndarray]
DistanceMetric = Callable[[np.ndarray, np.ndarray], float]
Population = List[Chromosome]

def changeSize(mat, coord1, coord2, scale):
  mat[coord1,coord2] = mat[coord1,coord2]*scale
  return mat


def w(x, matList):
  x_l, y_l = x.shape
  matList = [i.copy() for i in matList]
  #mat1 = mat1.copy()
  #mat2 = mat2.copy()
  #mat3 = mat3.copy()
  matList = [changeSize(changeSize(i, 0, 2, x_l), 1, 2, y_l) for i in matList]
  #mat1[0, 2] = mat1[0, 2] * x_l
  #mat2[0, 2] = mat2[0, 2] * x_l
  #mat3[0, 2] = mat3[0, 2] * x_l
  #mat1[1, 2] = mat1[1, 2] * y_l
  #mat2[1, 2] = mat2[1, 2] * y_l
  #mat3[1, 2] = mat3[1, 2] * y_l
  affList = [cv2.warpAffine(x, mat, x.shape) for mat in matList]
  #a = cv2.warpAffine(x, mat1, x.shape)
  #b = cv2.warpAffine(x, mat2, x.shape)
  #c = cv2.warpAffine(x, mat3, x.shape)
  return sum(affList)

# get sierpinski's image
def sierpinski():
  pic = np.ones((64,64), dtype = np.float32)
  mat1 = np.array([[0.5, 0, 0], 
                   [0, 0.5, 0]])
  mat2 = np.array([[0.5, 0, 0.5], 
                   [0, 0.5, 0]])
  mat3 = np.array([[0.5, 0, 0], 
                   [0, 0.5, 0.5]])
  for _ in range(100):
    pic = w(pic, [mat1, mat2, mat3])
  return pic
sierpinski = sierpinski()

def stacked_metric(pic1, pic2):
  d = 0
  size = pic1.shape[0]
  factor = 1
  while size >= 8:
    d_t = np.sum(np.square(pic1 - pic2))
    d += d_t * factor
    factor = 4 * factor ** 2
    size = int(size / 2)
    pic1 = cv2.resize(pic1, (size, size))
    pic2 = cv2.resize(pic2, (size, size))
  return np.exp(-d**2/10000**2)



def random_affine() -> np.ndarray:
    e, f = np.random.uniform(0., 1., 2)
    a, b = np.random.uniform(-e, 1-e, 2)
    if (a + b + e) > 1 or (a + b - e) < 0:
        a, b = a / 2, b / 2
    c, d = np.random.uniform(-f, 1-f, 2)
    if (c + d + f) > 1 or (c + d - f) < 0:
        c, d = c / 2, d / 2
    return np.array([[a, b, e],
                     [c, d, f]])

def det_abs(mat):
  return np.abs(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0])


# def fitness(args):
#   (mat1, mat2, mat3) = args
#   pic = np.ones((64,64), dtype = np.float32)
#   for _ in range(5):
#     pic = w(pic, mat1, mat2, mat3)
#   # cv2.imshow('pic', pic)
#   # cv2.waitKey(0)
#   return stacked_metric(sierpinski, pic)

def contFactor(mat):
  matValsSqrd = mat[0, 0]**2 + mat[1, 1]**2 + mat[0, 1]**2 + mat[1, 0]**2

  return np.sqrt((matValsSqrd + np.sqrt(matValsSqrd**2 - 4*((mat[0, 0] * mat[1, 1]) - (mat[0, 1] * mat[1, 0]))**2))/2)

def penalizeContFac(mats, STDCT):
 maxCFac = max([contFactor(mat) for mat in mats])
 assert (1 - maxCFac**10) >= 0
 return (1 - maxCFac**10) * np.exp(-(maxCFac/(2*STDCT))**2)

def penalizeCompFac(mats, STDCP):
  return(np.exp(-(len(mats)/(2*STDCP))**2))

# the better the closer to 0
def fitness(chromo, STDCT, STDCP):
   mats = chromo.genes
   y = w(sierpinski, mats)
  
   punish_on_identity_map = sum(det_abs(mat) for mat in mats) * 10000

   return stacked_metric(sierpinski, y) * penalizeContFac(mats, STDCT) * penalizeCompFac(mats, STDCP) #+ punish_on_identity_map 

def npGetArrInd(arr, item):
  for i in range(len(arr)):
    if np.array_equal(arr[i], item):
      return i
  raise NotFoundErr

""" if __name__ == '__main__':
  mat1 = random_affine()
  mat2 = random_affine()
  mat3 = random_affine()
  mat4 = np.array([[0.5, 0, 0],
                   [0, 0.5, 0]])
  mat5 = np.array([[0.5, 0, 0.5],
                    [0, 0.5, 0]])
  mat6 = np.array([[0.5, 0, 0],
                    [0, 0.5, 0.5]])

  chromo1 = (mat1, mat2, mat3)
  chromo2 = (mat4, mat5, mat6)
  print(chromo1)


  print(fitness(chromo1))
  print(fitness(chromo2)) """