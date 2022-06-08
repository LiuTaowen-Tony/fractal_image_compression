import cv2
import numpy as np

img = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128))
cv2.imwrite("lenna_128.png", img)

cv2.imshow("Original", img)
cv2.waitKey(0)