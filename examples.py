import cv2
import numpy as np

# a transformation consist of three 
def w(x, mat1, mat2, mat3):
  a = cv2.warpAffine(x, mat1, x.shape)
  b = cv2.warpAffine(x, mat2, x.shape)
  c = cv2.warpAffine(x, mat3, x.shape)
  return a + b + c

mat1 = np.array([[0.5, 0, 0], 
                  [0, 0.5, 0]])
mat2 = np.array([[0.5, 0, 32], 
                  [0, 0.5, 0]])
mat3 = np.array([[0.5, 0, 0], 
                  [0, 0.5, 32]])

def sierpinski():
  pic = np.ones((64,64), dtype = np.float32)
  for _ in range(100):
    pic = w(pic, mat1, mat2, mat3)

  return pic

sierpinski = sierpinski()

# a very small error applied on ifs (-0.01) on matrix
mat4 = np.array([[0.49, 0, 0],
                  [0, 0.49, 0]])
mat5 = np.array([[0.49, 0, 32],
                  [0, 0.49, 0]])
mat6 = np.array([[0.49, 0, 0],
                  [0, 0.49, 32]])  

def new_image():
  pic = np.ones((64,64), dtype = np.float32)
  for _ in range(100):
    pic = w(pic, mat4, mat5, mat6)

  return pic

sierpinski = np.uint8(sierpinski * 255)
print(sierpinski)
cv2.imwrite("sierpinski.png", sierpinski)

distorted = new_image()
distorted = np.uint8(distorted * 255)
cv2.imwrite("distorted.png", distorted)

# cv2.imshow('sierpinski', sierpinski)

# cv2.imshow('new_image', new_image())
# cv2.waitKey(0)