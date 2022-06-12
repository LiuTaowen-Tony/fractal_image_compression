import cv2
import numpy as np
from scipy import optimize

# a transformation consist of three 
def w(x, mat1, mat2, mat3):
  x_l, y_l = x.shape
  mat1 = mat1.copy()
  mat2 = mat2.copy()
  mat3 = mat3.copy()
  mat1[0, 2] = mat1[0, 2] * x_l
  mat2[0, 2] = mat2[0, 2] * x_l
  mat3[0, 2] = mat3[0, 2] * x_l
  mat1[1, 2] = mat1[1, 2] * y_l
  mat2[1, 2] = mat2[1, 2] * y_l
  mat3[1, 2] = mat3[1, 2] * y_l
  a = cv2.warpAffine(x, mat1, x.shape)
  b = cv2.warpAffine(x, mat2, x.shape)
  c = cv2.warpAffine(x, mat3, x.shape)
  return a + b + c


# using the rms metric since this is the only sensible metric for 
# computerized images
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
  return d

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
    pic = w(pic, mat1, mat2, mat3)
  return pic
sierpinski = sierpinski()

cv2.imwrite("sierpinski.tiff", sierpinski)


def args_to_mats(args):
  mat1 = args[0:6].reshape((2, 3))
  mat2 = args[6:12].reshape((2, 3))
  mat3 = args[12:18].reshape((2, 3)) 
  return (mat1, mat2, mat3)

def mats_to_args(mats):
  mat1 = mats[0].flatten()
  mat2 = mats[1].flatten()
  mat3 = mats[2].flatten()
  return np.concatenate((mat1, mat2, mat3))

mat1 = np.array([[0.5, 0, 0], 
                  [0, 0.5, 0]])
mat2 = np.array([[0.5, 0, 0.5], 
                  [0, 0.5, 0]])
mat3 = np.array([[0.5, 0, 0], 
                  [0, 0.5, 0.5]])

# want to minimize
# d(S, A(S) + B(S) + C(S))
identity_map = np.array([[1, 0, 0],
                         [0, 1, 0]], dtype=np.float32)

def det(mat):
  return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]

# d(S, w(S))
def cost(args):
  (mat1, mat2, mat3) = args_to_mats(args)
  y = w(sierpinski, mat1, mat2, mat3)
  
  punish_on_identity_map = 0
  punish_on_identity_map += det(mat1) * 5000
  punish_on_identity_map += det(mat2) * 5000
  punish_on_identity_map += det(mat3) * 5000

  return stacked_metric(sierpinski, y) + punish_on_identity_map 

# sanity check, the correct parameter should give 0 cost

print(cost(mats_to_args((mat1, mat2, mat3))))

# mat4 = identity_map.copy() + np.random.randn(2, 3) * 1

mat4 = np.random.randn(2, 3)
mat5 = np.random.randn(2, 3)
mat6 = np.random.randn(2, 3)

mat7 = identity_map - 0.1

print(mat4)

print(cost(mats_to_args((mat4, mat5, mat6))))
print(cost(mats_to_args((mat7, mat7, mat7))))

# minimize 
# d(S, w(w(w...(w(random_image)))))
def cost2(args):
  (mat1, mat2, mat3) = args_to_mats(args)
  pic = np.ones((64,64), dtype = np.float32)
  for _ in range(10):
    pic = w(pic, mat1, mat2, mat3)
  return stacked_metric(sierpinski, pic)

# bounds for parameters
bounds = [(0, 1)] * 18
bounds = np.array(bounds)

guess1 = mat1 + np.random.randn(2, 3) * 0.1
guess2 = mat2 + np.random.randn(2, 3) * 0.1
guess3 = mat3 + np.random.randn(2, 3) * 0.1


# actually run the minimization algorithm
# res = optimize.minimize(cost2, guess, bounds = bounds, options={'maxiter': 100000})
res = optimize.dual_annealing(cost, bounds = bounds, maxiter = 100000)
# res = optimize.minimize(cost, mats_to_args((guess1, guess2, guess3)), bounds = bounds, options={'maxiter': 100000})
mats = (args_to_mats(res.x))

# print the result
print(res)
for i in args_to_mats(res.x):

  print(np.array(i).reshape((2, 3)))

# show the image applied to sierpinski (w(sierpinski))
cv2.imshow("A(S) + B(S) + C(S)", w(sierpinski, *mats))
cv2.waitKey(0)

# show encoding a single image using the result (encoded ifs)
pic = np.ones((128,128), dtype = np.float32)
for _ in range(10):
  pic = w(pic, *mats)
cv2.imshow("result", pic)
cv2.waitKey(0)
