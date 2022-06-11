import cv2
import numpy as np
from scipy import optimize

# a transformation consist of three 
def w(x, mat1, mat2, mat3):
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
    factor = 8 * factor ** 2
    size = int(size / 2)
  return d

# get sierpinski's image
def sierpinski():
  pic = np.ones((64,64), dtype = np.float32)
  mat1 = np.array([[0.5, 0, 0], 
                   [0, 0.5, 0]])
  mat2 = np.array([[0.5, 0, 32], 
                   [0, 0.5, 0]])
  mat3 = np.array([[0.5, 0, 0], 
                   [0, 0.5, 32]])
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
mat2 = np.array([[0.5, 0, 32], 
                  [0, 0.5, 0]])
mat3 = np.array([[0.5, 0, 0], 
                  [0, 0.5, 32]])

# want to minimize
# d(S, A(S) + B(S) + C(S))
def cost(args):
  (mat1, mat2, mat3) = args_to_mats(args)
  y = w(sierpinski, mat1, mat2, mat3)
  return stacked_metric(sierpinski, y) 

# sanity check, the correct parameter should give 0 cost
assert cost(mats_to_args((mat1, mat2, mat3))) == 0

# minimize 
# d(S, w(w(w...(w(random_image)))))
def cost2(args):
  (mat1, mat2, mat3) = args_to_mats(args)
  pic = np.ones((64,64), dtype = np.float32)
  for _ in range(10):
    pic = w(pic, mat1, mat2, mat3)
  return stacked_metric(sierpinski, pic)

# bounds for parameters
bounds = [(0,1), (0, 1), (0, 64), (0, 1), (0, 1), (0, 64)] * 3
bounds = np.array(bounds)

# initial guess, this is actually very close to the correct answer
# but it still doesn't work well
guess_mat1 = np.array([[0.5, 0, 0], 
                        [0, 0.5, 0]])
guess_mat2 = np.array([[0.5, 0, 32],
                        [0, 0.5, 0]])
guess_mat3 = np.array([[0.5, 0, 0],
                        [0, 0.5, 32]])
guess = np.concatenate((guess_mat1.flatten(), guess_mat2.flatten(), guess_mat3.flatten()))
guess = guess + np.random.rand(guess.shape[0])

print(guess)

# actually run the minimization algorithm
# res = optimize.minimize(cost2, guess, bounds = bounds, options={'maxiter': 100000})
res = optimize.dual_annealing(cost2, bounds = bounds, maxiter = 10000)
mats = (args_to_mats(res.x))

# print the result
print(res)
for i in args_to_mats(res.x):
  print(np.array(i).resize((2, 3)))

# show the image applied to sierpinski (w(sierpinski))
cv2.imshow("A(S) + B(S) + C(S)", w(sierpinski, *mats))
cv2.waitKey(0)

# show encoding a single image using the result (encoded ifs)
pic = np.ones((128,128), dtype = np.float32)
for _ in range(10):
  pic = w(pic, *mats)
cv2.imshow("result", pic)
cv2.waitKey(0)
