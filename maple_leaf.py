import numpy as np
import cv2

# grey scale affines
def w(x, transformations):
  x_l, y_l = x.shape
  # contrast in (0, 1), brightness in (0, 1)
  result = np.zeros(x.shape, dtype = np.uint8)
  for (mat, contrast, brightness) in transformations:
    brightness = int(brightness * 255)
    mat = mat.copy()
    mat[0,2] = mat[0,2] * x_l
    mat[1,2] = mat[1,2] * y_l
    t = np.uint8(x * contrast)
    # to prevent overflow
    t = t + np.minimum(255 - t, brightness)
    t = cv2.warpAffine(t, mat, x.shape)
    result = result + np.minimum(255 - result, t)
  return result

def maple_leaf(_a, _b, _c, _d):
  mat1 = np.array([[0.6, 0, 0.2],
                    [0, 0.6, 0]])
  mat2 = np.array([[0.6, 0, 0.2],
                    [0, 0.6, 0.5]])
  c = np.cos(np.pi / 4) * 0.6
  s = np.sin(np.pi / 4) * 0.6
  mat3 = np.array([[c, s, -0.18],
                   [-s, c, 0.67]])
  mat4 = np.array([[c, -s, 0.8],
                   [s, c, 0.28]])

  pic = np.ones((128,128), dtype = np.uint8) * 255
  for _ in range(6):
    # pic = w(pic, [(mat3, *_c)])
    pic = w(pic, [(mat1, *_a), (mat2, *_b), (mat3, *_c), (mat4, *_d)])
    # pic = w(pic, [(mat1, *_a), (mat2, *_b), (mat3, *_c)])

  return pic

maple_leaf_white = maple_leaf((1, 0), (1, 0), (1, 0), (1, 0))

maple_leaf_white_0_1 = np.float32(maple_leaf_white) / 255

# cv2.imshow("leaf", maple_leaf_white_0_1)
# cv2.waitKey(0)
