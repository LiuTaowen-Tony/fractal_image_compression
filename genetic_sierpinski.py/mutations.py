import numpy as np
import cv2

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
  return mat_vec_to_affine(mat, vec)

def scale(w):
  # probably need to change this
  s = np.random.uniform(0.5, 1.5)
  w_copy = w.copy()
  (a, b, e,
   c, d, f) = w_copy.reshape((6,))
  if np.random.random() < 0.5:
    w_copy[0, 0] = a * s
  else:
    w_copy[0, 1] = b * s
  return w_copy

def skew(w):
  s = np.random.uniform(0.0, 1.0)
  w_copy = w.copy()
  (a, b, e,
   c, d, f) = w_copy.reshape((6,))
  if np.random.random() < 0.5:
    w_copy[1, 0] = a * s + b
    w_copy[1, 1] = c * s + d
  else:
    w_copy[0, 0] = a + b * s
    w_copy[0, 1] = c + d * s
  return w_copy

def translation( w):
  (a, b, e,
   c, d, f) = w_copy.reshape((6,))
  r = (e + f) / 4
  x = np.random.uniform(-r, r)
  w_copy = w.copy()
  if np.random.random() < 0.5:
    w_copy[0, 2] = e + x
  else:
    w_copy[1, 2] = f + x
  return w_copy



    




