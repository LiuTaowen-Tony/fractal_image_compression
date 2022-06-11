import numpy as np
import cv2
from typing import List, Tuple, Function

Chromosome = List[np.ndarray]
DistanceMetric = Function[(np.ndarray, np.ndarray), float]
Population = List[Chromosome]

def similarity_measure(pic1, pic2):
  assert pic1.shape == pic2.shape
  assert pic1.dtype == pic2.dtype == np.bool8
  intersection = np.logical_and(pic1, pic2)
  union = np.logical_or(pic1, pic2)
  return np.sum(intersection) / np.sum(union)

def test_similarity_measure():
  a = np.random.randint(0, 2, size = (64, 64), dtype = np.bool8)
  assert 1 == similarity_measure(a, a)
  a_sum = a.sum()
  assert a_sum / (64 * 64) == similarity_measure(a, np.ones_like(a, dtype = np.bool8))
  print("similarity_measure test passed")

def sierpinski_gen(length):
  mat1 = np.array([[0.5, 0, 0], 
                   [0, 0.5, 0]])
  mat2 = np.array([[0.5, 0, 0.5], 
                   [0, 0.5, 0]])
  mat3 = np.array([[0.5, 0, 0], 
                   [0, 0.5, 0.5]])
  chromesome = [mat1, mat2, mat3]
  pic = np.ones((length,length), dtype = np.bool8)
  for _ in range(int(np.log(length) * 3)):
    pic = dilate(np.uint8(pic) * 255)
    pic = np.bool8(pic)
    pic = apply_transformations(pic, chromesome)
  return pic

sierpinski_128 = sierpinski_gen(128)

def apply_transformations(pic, chromosome : List[np.ndarray]):
  new_pic = np.zeros(pic.shape, dtype = np.uint8)
  pic_copy = np.copy(pic)
  pic_copy = np.uint8(pic_copy)
  pic_copy *= 255
  len_x, len_y = pic.shape
  for gene in chromosome:
    mat = gene.reshape((2, 3)).copy()
    mat[0,2] = mat[0, 2] * len_x
    mat[1,2] = mat[1, 2] * len_y
    new_pic += cv2.warpAffine(pic_copy, mat, new_pic.shape)
  new_pic = np.bool8(new_pic)
  return new_pic

def dilate(pic):
  kernel = np.ones((3, 3), dtype = np.uint8)
  return cv2.dilate(pic, kernel)

def test_apply_transformations():
  a = np.random.randint(0, 2, size = (64, 64), dtype = np.bool8)
  chromesome = [np.array([[0.5, 0, 0],
                          [0, 0.5, 0]])]
  b = apply_transformations(a, chromesome)
  cv2.imshow("b", np.uint8(b) * 255)
  cv2.waitKey(0)
  print("apply_transformations test passed")


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

if "__main__" == __name__:
  test_apply_transformations()
  test_similarity_measure()
  sierpinski = sierpinski_gen(2000)
  cv2.imshow("sierpinski", np.uint8(sierpinski) * 255)
  cv2.waitKey(0)
