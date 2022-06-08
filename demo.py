import cv2
from typing import Tuple
from itertools import product

from numpy import block

def get_subscripts(domain_position : Tuple[int, int], 
                   range_position : Tuple[int, int], block_size=(4,4)):
  sx, sy = block_size
  dsx, dsy = domain_position[0] * sx, domain_position[1] * sy
  rsx, rsy = range_position[0] * sx, range_position[1] * sy
  dx = (dsx + i for i in range(sx))
  rx = (rsx + i for i in range(sx))
  dy = (dsy + i for i in range(sy))
  ry = (rsy + i for i in range(sy))
  return product(dx, rx, dy, ry)

def distance(domain_image, range_image, subscripts):
  distance = 0
  for dx, rx, dy, ry in subscripts:
      distance += (domain_image[dx, dy] - range_image[rx, ry]) ** 2
  return distance

def determine_alpha(domain_image, range_image,subscripts, block_size=(4,4)):
  sx, sy = block_size
  fst_term = 0
  for dx, rx, dy, ry in subscripts:
      fst_term += domain_image[dx, dy] * range_image[rx, ry]
  fst_term *= sx * sy

  sum_domain = 0
  for dx, rx, dy, ry in subscripts:
      sum_domain += domain_image[dx, dy]

  sum_range = 0
  for dx, rx, dy, ry in subscripts:
      sum_range += range_image[rx, ry]


  sum_domain_sq = 0
  for dx, rx, dy, ry in subscripts:
      sum_domain_sq += domain_image[dx, dy] ** 2

  numerator = fst_term - sum_domain * sum_range
  denominator = sx * sy * sum_domain_sq - sum_domain ** 2
  return numerator / denominator

def detemine_t0(domain_image, range_image, subscripts, block_size=(4,4)):
  sx, sy = block_size
  sum_domain = 0
  for dx, rx, dy, ry in subscripts:
      sum_domain += domain_image[dx, dy]

  sum_range = 0
  for dx, rx, dy, ry in subscripts:
      sum_range += range_image[rx, ry]

  sum_domain_sq = 0
  for dx, rx, dy, ry in subscripts:
      sum_domain_sq += domain_image[dx, dy] ** 2 

  return (sum_domain ** 2 - sum_range ** 2) / (sx * sy * sum_domain_sq - sum_domain ** 2)

def find_block_book(domain_image, range_image, block_size=(4,4)):
  sx, sy = block_size
  dsize_x, dsize_y = domain_image.shape
  rsize_x, rsize_y = range_image.shape
  # should be (128, 128) and (64, 64)
  dblock_size_x, dblock_size_y = (dsize_x // sx, dsize_y // sy)
  rblock_size_x, rblock_size_y = (rsize_x // sx, rsize_y // sy)
  # should be (32, 32) and (16, 16)
  block_book = {}
  transformed = np.zeros(block_size)
  for domain_block_position in product(range(dblock_size_x), range(dblock_size_y)):
    min_distance = 1e10
    min_range_block_postion_a_t0 = None
    for range_block_position in product(range(rblock_size_x), range(rblock_size_y)):
      block_subscripts = get_subscripts(domain_block_position, range_block_position, block_size)
      a = determine_alpha(domain_image, range_image, block_subscripts)
      t0 = detemine_t0(domain_image, range_image, block_subscripts)
      (dsx, rsx, dsy, rsy) = block_subscripts[0]
      for i, j in product(range(sx), range(sy)):
        transformed[i, j] = a * domain_image[dsx + i, dsy + j] + t0
      distance = 0
      for i, j in product(range(sx), range(sy)):
        distance += (transformed[i, j] - range_image[rsx + i, rsy + j]) ** 2
      if min_distance == 0 or distance < min_distance:
        min_distance = distance
        min_range_block_postion_a_t0 = (range_block_position, a, t0)
    if min_range_block_postion_a_t0 in block_book:
      block_book[domain_block_position].append(min_range_block_postion_a_t0)
    else:
      block_book[domain_block_position] = [min_range_block_postion_a_t0]

  return encode(range_image, block_book, block_size)

def encode(range_image, block_book, block_size):
  sx, sy = block_size
  dsize_x, dsize_y = range_image.shape * 2
  rsize_x, rsize_y = range_image.shape
  dblock_size_x, dblock_size_y = (dsize_x // sx, dsize_y // sy)
  rblock_size_x, rblock_size_y = (rsize_x // sx, rsize_y // sy)
  encoded_image = np.zeros((dsize_x, dsize_y))
  for domain_block_position in product(range(dblock_size_x), range(dblock_size_y)):
    for range_block_position, a, t0 in block_book[domain_block_position]:
      block_subscripts = get_subscripts(domain_block_position, range_block_position, block_size)
      (dsx, rsx, dsy, rsy) = block_subscripts[0]
      for i, j in product(range(sx), range(sy)):
        encoded_image[dsx + i, dsy + j] = a * range_image[rsx + i, rsy + j] + t0
  return encoded_image
      