#!/usr/bin/env python
"""

"""
__version__ = "0.1"
import cv2
import imutils
import math
import logging
import numpy as np
import re
import time
from matplotlib import pyplot as plt


def show_img(imgs, titles, delay):
  for (i,img) in enumerate(imgs):
    cv2.imshow(titles[i], img)
  cv2.waitKey(delay)
  cv2.destroyAllWindows()

def compare_shape(img, dim):
  """ damn cv2 img size doesn't match dim"""
  return (img.shape[1] == dim[0] and img.shape[0] == dim[1])

def sliding_window(image, window_size, step_size):
  for y in xrange(0, image.shape[0], step_size[1]):
    for x in xrange(0, image.shape[1], step_size[0]):
      yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def find_surrounding_box(image, w, h):
  """ create a new image and draw "image in the center"""
  if image.shape[0] >= h and image.shape[1] >= w:
    return image
  height = max(h, image.shape[0])
  width = max(w, image.shape[1])
  dst_image = np.zeros((height, width), dtype='uint8')
  offset_y = (height-image.shape[0])/2; offset_x = (width-image.shape[1])/2
  dst_image[offset_y:(offset_y + image.shape[0]), offset_x:(offset_x + image.shape[1])] = image
  return dst_image

def shrink(image, width, height, inter=cv2.INTER_AREA):
  #create a new box with the desired dimension and draw the re_sized image
  dim = (width, height)
  (h, w) = image.shape[:2]
  aspect_ratio = float(w)/float(h)
  assert w > width or h > height
  if float(width)/float(height) > aspect_ratio:
    new_w = int(float(height) * aspect_ratio)
    new_h = height
  else:
    new_h = int(float(width)/aspect_ratio)
    new_w = width
  dim = (new_w, new_h)
  image = cv2.resize(image, dim, interpolation=inter)
  return find_surrounding_box(image, width, height)

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
  """ resize and make sure new height and width are within limits"""
  dim = (width, height)
  (h, w) = image.shape[:2]
  # if both the width and height are None, then return the # original image
  if width is None and height is None:
    return image
  #check to see if the width is None
  aspect_ratio = float(w)/float(h)
  #calculate new width, height w/ot considering limits
  new_w = width if width is not None else int(height * aspect_ratio)
  if width is not None and height is not None:
    #both are present. create a new box with the desired dimension and draw the re_sized image
    if float(width)/float(height) > aspect_ratio:
      new_h = int(float(width)/aspect_ratio)
      new_w = width
    else:
      new_w = int(float(height) * aspect_ratio)
      new_h = height
  elif width is not None:
    new_h = int(float(width)/aspect_ratio)
    new_w = width
  else:
    new_h = height
    new_w = int(float(height) * aspect_ratio)
  dim = (new_w, new_h)
  # resize the image
  return cv2.resize(image, dim, interpolation=inter)

def pyramid(image, scale=1.5, minSize=(30, 30), reversed=False):
  # yield the original image
  (orig_h, orig_w) = image.shape
  if reversed:
    #image = shrink(image, width=minSize[0], height=minSize[1], inter=cv2.INTER_LINEAR)
    image = resize(image, width=minSize[0], height=minSize[1], inter=cv2.INTER_LINEAR)
  yield image
  # keep looping over the pyramid
  while True:
    # compute the new dimensions of the image and resize it
    if reversed:
      w = int(image.shape[1] * scale)
      image = resize(image, width=w, inter=cv2.INTER_LINEAR)
    else:
      w = int(image.shape[1] / scale)
      h = int(image.shape[0] / scale)
      #image = shrink(image, w, h)
      image = resize(image, w, h)

    # if the resized image does not meet the supplied minimum
    # size, then stop constructing the pyramid
    if image.shape[0] < minSize[1] and image.shape[1] < minSize[0]:
      break
    if image.shape[0] > orig_h or image.shape[1] > orig_w:
      break

    if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
      image = find_surrounding_box(image, minSize[0], minSize[1])

    # yield the next image in the pyramid
    yield image

def rotate(image, angle):
  """Rotate image 'angle' degrees.
  http://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c/22042434#22042434
  How it works:
    - Creates a blank image that fits any rotation of the image. To achieve
      this, set the height and width to be the image's diagonal.
    - Copy the original image to the center of this blank image
    - Rotate using warpAffine, using the newly created image's center
      (the enlarged blank image center)
    - Translate the four corners of the source image in the enlarged image
      using homogenous multiplication of the rotation matrix.
    - Crop the image according to these transformed corners
  """
  if angle == 0:
    return image

  diagonal = int(math.sqrt(pow(image.shape[0], 2) + pow(image.shape[1], 2)))
  offset_x = (diagonal - image.shape[0]) / 2
  offset_y = (diagonal - image.shape[1]) / 2
  if len(image.shape) > 2:
    dst_image = np.zeros((diagonal, diagonal, image.shape[2]), dtype='uint8')
  else:
    dst_image = np.zeros((diagonal, diagonal), dtype='uint8')
  image_center = (diagonal / 2, diagonal / 2)

  R = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  if len(image.shape) > 2:
    dst_image[offset_x:(offset_x + image.shape[0]), offset_y:(offset_y + image.shape[1]), :] = image
  else:
    dst_image[offset_x:(offset_x + image.shape[0]), offset_y:(offset_y + image.shape[1])] = image
  dst_image = cv2.warpAffine(dst_image, R, (diagonal, diagonal), flags=cv2.INTER_LINEAR)

  # Calculate the rotated bounding rect
  x0 = offset_x
  x1 = offset_x + image.shape[0]
  x2 = offset_x
  x3 = offset_x + image.shape[0]

  y0 = offset_y
  y1 = offset_y
  y2 = offset_y + image.shape[1]
  y3 = offset_y + image.shape[1]

  corners = np.zeros((3, 4))
  corners[0, 0] = x0
  corners[0, 1] = x1
  corners[0, 2] = x2
  corners[0, 3] = x3
  corners[1, 0] = y0
  corners[1, 1] = y1
  corners[1, 2] = y2
  corners[1, 3] = y3
  corners[2:] = 1

  c = np.dot(R, corners)

  x = int(c[0, 0])
  y = int(c[1, 0])
  left = x
  right = x
  up = y
  down = y

  for i in range(4):
    x = int(c[0, i])
    y = int(c[1, i])
    if (x < left): left = x
    if (x > right): right = x
    if (y < up): up = y
    if (y > down): down = y
  h = down - up
  w = right - left

  if len(image.shape) > 2:
    cropped = np.zeros((w, h, 3), dtype='uint8')
    cropped[:, :, :] = dst_image[left:(left + w), up:(up + h), :]
  else:
    cropped = np.zeros((w, h), dtype='uint8')
    cropped[:, :] = dst_image[left:(left + w), up:(up + h)]
  return cropped
