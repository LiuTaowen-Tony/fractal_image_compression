import numpy as np
import cv2


# grey scale affines
def w(x, transformations):
  x_l, y_l = x.shape
  # contrast in (0, 1), brightness in (0, 255)
  result = np.zeros(x.shape, dtype = np.uint8)
  for (mat, contrast, brightness) in transformations:
    mat = mat.copy()
    mat[0,2] = mat[0,2] * x_l
    mat[1,2] = mat[1,2] * y_l
    t = np.uint8(x * contrast)
    # to prevent overflow
    t = t + np.minimum(255 - t, brightness)
    t = cv2.warpAffine(t, mat, x.shape)
    # print(t)
    result = result + np.minimum(255 - result, t)
    # cv2.imshow("t", t)
    # cv2.waitKey(0)
    # cv2.imshow('result', result)
    # cv2.waitKey(0)
  return result

def grey_sierpinski(a, b, c):
  pic = np.ones((64,64), dtype = np.uint8) * 255
  mat1 = np.array([[0.5, 0, 0], 
                   [0, 0.5, 0]])
  mat2 = np.array([[0.5, 0, 0.5], 
                   [0, 0.5, 0]])
  mat3 = np.array([[0.5, 0, 0], 
                   [0, 0.5, 0.5]])
  for _ in range(6):
    pic = w(pic, [(mat1, *a), (mat2, *b), (mat3, *c)])
  return pic

a = (0.5, 50)
b = (0.9, 100)
c = (0.9, 0)

r = grey_sierpinski(a, b, c)
g = grey_sierpinski(b, c, a)
b = grey_sierpinski(c, a, b)

final_img = np.dstack([r, g, b])
print(final_img.shape)
cv2.imshow("r", r)
cv2.imshow("g", g)
cv2.imshow("b", b)
cv2.imshow('final_img', final_img)

cv2.waitKey(0)
