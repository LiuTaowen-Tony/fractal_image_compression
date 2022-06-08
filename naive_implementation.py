from itertools import chain
import numpy as np
import cv2
from typing import List


def rms(pic1, pic2):
  return np.sum(np.abs(pic1 - pic2) ** 2)

def determine_params(domain_block, range_block):
  assert domain_block.shape == range_block.shape == (4,4)
  domain_sum = np.sum(domain_block)
  domain_square_sum = np.sum(domain_block ** 2)
  range_sum = np.sum(range_block)
  range_square_sum = np.sum(range_block ** 2)

  domain_range_prod_sum = np.sum(domain_block * range_block)

  a_numer = (4 ** 2) * domain_range_prod_sum - domain_sum * range_sum
  denom = (4 ** 2) * domain_square_sum - domain_sum ** 2
  a = a_numer / denom

  t0_numer = domain_sum ** 2 - range_sum ** 2
  t0 = t0_numer / denom
  return a, t0

def apply_transformation(domain_block, a, t0):
  assert domain_block.shape == (4,4)
  result = a * domain_block + t0
  return result

def make_domain_partition(pic):
  assert pic.shape == (64,64)
  x, y = pic.shape
  block_x_num = x // 4
  block_y_num = y // 4
  domain_blocks = []
  for i in range(block_x_num):
    line = []
    for j in range(block_y_num):
      line.append(pic[i*4:(i+1)*4, j*4:(j+1)*4])
    domain_blocks.append(line)
  return domain_blocks

def make_range_partition(pic):
  assert pic.shape == (128,128)
  x, y = pic.shape
  block_x_num = x // 4
  block_y_num = y // 4
  range_blocks = []
  for i in range(block_x_num):
    line = []
    for j in range(block_y_num):
      line.append(pic[i*4:(i+1)*4, j*4:(j+1)*4])
    range_blocks.append(line)
  return range_blocks

def find_code_book(domain_blocks, range_blocks : List[List[np.ndarray]]):
  code_book = {}
  for rl, r_line_blk in enumerate(range_blocks):
    for cl, range_block in enumerate(r_line_blk):
      min_distance = 9999999999999999999
      for line, line_blk in enumerate(domain_blocks):
        for column, domain_block in enumerate(line_blk):
          a, t0 = determine_params(domain_block, range_block)
          c = apply_transformation(domain_block, a, t0)
          distance = rms(c, range_block)
          if distance < min_distance:
            min_distance = distance
            min_index = (line, column)
            min_params = (a, t0)
      code_book[(rl, cl)] = (min_index, min_params)
  return code_book


def encode(initial_image, code_book):
  assert initial_image.shape == (128,128)
  x, y = initial_image.shape
  initial_image = cv2.resize(initial_image, (64,64))
  encoded_image = np.zeros((x, y), dtype = np.float32)
  domain_blocks = make_domain_partition(initial_image)
  print(initial_image)
  for (rl, cl), ((dl, dc), (a, t0)) in code_book.items():
    encoded_image[rl*4:(rl+1)*4, cl*4:(cl+1)*4] = apply_transformation(domain_blocks[dl][dc], a, t0)
  encoded_image[encoded_image > 1.0] = 1.0
  encoded_image[encoded_image < 0.0] = 0.0
  return encoded_image

def find_code_book_n_times(to_be_encoded, n):
  code_books = []
  encoded = to_be_encoded
  range_blocks = make_range_partition(to_be_encoded)
  for _ in range(n): 
    domain_blocks = make_domain_partition(cv2.resize(encoded, (64,64)))
    code_book = find_code_book(domain_blocks, range_blocks)
    encoded = encode(encoded, code_book)
    code_books.append(code_book)
  return code_books

def encode_with_n_code_books(initial_image, code_books):
  encoded = initial_image
  for code_book in code_books:
    encoded = encode(encoded, code_book)
  return encoded

def main_one_book():
  img = cv2.imread("lenna_128.png", cv2.IMREAD_GRAYSCALE)
  img = np.float32(img)
  img /= 255
  print(img)
  domain_blocks = make_domain_partition(cv2.resize(img, (64,64)))
  range_blocks = make_range_partition(img)
  code_book = find_code_book(domain_blocks, range_blocks)
  print(code_book)
  new_img = np.ones((128,128), dtype = np.float32)
  for _ in range(10):
    cv2.imshow("",new_img)
    cv2.waitKey(0)
    new_img = encode(new_img, code_book)

def main_n_books():
  img = cv2.imread("lenna_128.png", cv2.IMREAD_GRAYSCALE)
  img = cv2.resize(img, (128,128))
  img = np.float32(img)
  img /= 255
  print(img)
  code_books = find_code_book_n_times(img, 10)
  for i in range(10):
    new_img = np.ones((128,128), dtype = np.float32)
    new_img = encode_with_n_code_books(new_img, code_books[:i+1])
    new_img *= 255
    new_img = np.uint8(new_img)
    cv2.imwrite("encoded_lenna" + str(i) + ".png", new_img)

main_n_books()
