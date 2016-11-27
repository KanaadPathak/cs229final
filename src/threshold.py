#!/usr/bin/env python
"""

"""
__version__ = "0.2"
import argparse, sys, os
import cv2
import image_feature
import imgutil
import logging
import numpy as np
import json
import random
import re
import xml.etree.ElementTree as ET
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def showImages(imgs,titles):
  ''' show images with tilels '''
  for i in range(0,4):
    if len(imgs[i].shape) == 3:
      converted = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
      plt.subplot(141+i),plt.imshow(converted),plt.title(titles[i])
    else:
      plt.subplot(141+i),plt.imshow(imgs[i], cmap='gray'),plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
  plt.show()


def kmeans_threshold(img, visualize=False):
  img = cv2.imread(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  orig = img.copy()

  (h,w,d) = img.shape
  img = img.reshape((h * w, d))
  clt = KMeans(2)
  clt.fit(img)

  labels = clt.predict(img)

  for i in range(len(labels)):
    if labels[i]  == labels[0]:
      img[i] = [0, 0, 0]

  print "done, drawing..."
  img = img.reshape((h, w, d))
  cv2.imshow('original', orig)
  cv2.imshow('threshold', img)
  cv2.waitKey(0)

def filtering(img_gray,esp):
  if esp == "median":
    return cv2.medianBlur(img_gray,5)
  elif esp == "gaussian":
    return cv2.GaussianBlur(img_gray,(5,5),0)
  elif esp == "bilateral":
    return cv2.bilateralFilter(img_gray,5,50,100)
  elif esp == "adaptive":
    return cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 0)



'''
Find the contours of a canny image, returns an image with the 3 largest contours,
a bounding box around them, th bounding box, the contours and perimeters sorted
'''
def largestContours(canny,img,img_gray, visualize = False):
  # Perform morphology
  #se = np.ones((7, 7), dtype='uint8')
  #canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, se)

  canny, contours, hierarchy = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
  img_contour = np.copy(img) # Contours change original image.
  #cv2.drawContours(img_contour, contours, -1, (0,255,0), 3) # Draw all - For visualization only

  # Contours -  maybe the largest perimeters pinpoint to the leaf?
  perimeter = []
  max_perim = [0,0]
  i = 0

  # Find perimeter for each contour i = id of contour
  for each_cnt in contours:
    prm = cv2.arcLength(each_cnt,False)
    perimeter.append([prm,i])
    i += 1

  # Sort them
  perimeter = sorted(perimeter, key=lambda x: x[0], reverse = True)

  unified = []
  max_index = []
  # Draw max contours
  if len(perimeter) == 0:
    logging.error("Cannot find contours!")
    return (None, None, None, None )

  #pick = max(int(len(perimeter) * 0.1 + 1), min(len(perimeter), 3))
  pick = min(len(perimeter), 3)
  logging.debug(" found contours: %d use: %d", len(perimeter), pick)
  for i in range(0, pick):
    index = perimeter[i][1]
    max_index.append(index)
    if visualize:
      cv2.drawContours(img_contour, contours, index, (255,0,0), 3)

  # Get convex hull for max contours and draw them
  cont = np.vstack(contours[i] for i in max_index)
  hull = cv2.convexHull(cont)
  unified.append(hull)
  if visualize:
    cv2.drawContours(img_contour,unified,-1,(0,0,255),3)

  return img_contour, contours, perimeter, hull


'''
Given a convex hull apply graph cut to the image
Assumptions:
- Everything inside convex hull is the foreground object - cv2.GC_FGD or 1
- Everything outside the rectangle is the background -  cv2.GC_BGD or 0
- Between the hull and the rectangle is probably foreground - cv2.GC_PR_FGD or 3
'''
def cut_graph_from_hull(hull,img):
  # First create our rectangle that contains the object
  y_corners = np.amax(hull, axis=0)
  x_corners = np.amin(hull,axis=0)
  x_min = x_corners[0][0]
  x_max = x_corners[0][1]
  y_min = y_corners[0][0]
  y_max = y_corners[0][1]

  # input/output mask
  mask = np.ones(img.shape[:2],np.uint8) * cv2.GC_PR_BGD

  # Values needed for algorithm
  bgdModel = np.zeros((1,65),np.float64)
  fgdModel = np.zeros((1,65),np.float64)

  #method #1 use GC_INIT_WITH_RECT
  #rect = (x_min,x_max,y_min,y_max)
  #cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
  #method #2 use GC_INIT_WITH_MASK
  rect = None
  contours = [hull]
  cv2.drawContours(mask, contours,-1, (cv2.GC_PR_FGD,cv2.GC_PR_FGD,cv2.GC_PR_FGD) ,-1)
  cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

  #apply mask to the image
  mask2 = np.where((mask==cv2.GC_PR_BGD)|(mask==cv2.GC_BGD),0,1).astype('uint8')
  img = img*mask2[:,:,np.newaxis]
  return img

def auto_canny(image, sigma=0.33):
  # compute the median of the single channel pixel intensities
  #canny = cv2.Canny(filtered,100,200)
  #canny_unfiltered = cv2.Canny(img_gray,100,200)
  v = np.median(image)
  # apply automatic Canny edge detection using the computed median
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edged = cv2.Canny(image, lower, upper)

  # return the edged image
  return edged

def remove_background(orig, visualize=False):
  """ simple technique to remove background"""
  #img = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
  img = orig
  img_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
  #blur
  filtered = filtering(img_gray,"gaussian")
  #detect edge
  canny = auto_canny(filtered)
  #canny_unfiltered = auto_canny(img_gray)

  ## SEGMENTATION by finding largest Contour - Not the best segmentation
  img_contour, contours, perimeters, hull = largestContours(canny,img,img_gray, visualize)
  if img_contour is None:
    logging.error("hitting error, return original image")
    return orig

  # Grabcut - Same bounding box than contours...
  img_grcut = cut_graph_from_hull(hull,img)

  ## Show images
  images = [img, canny, img_contour, img_grcut ]
  titles = ["original", "edges", "largest contour", "graph cut"]
  if visualize:
    showImages(images, titles)
  return img_grcut


def process(img_path, output_path, visualize=False):
  for subdir, dirs, files in os.walk(img_path):
    for f in files:
      m = re.match(r"(.*)[.]jpg", f)
      if m is None:
        continue
      filename = os.path.join(subdir,f)
      logging.info("working on %(filename)s" % locals())
      img = cv2.imread(filename)
      img = remove_background(img, visualize)
      output_filename = os.path.join(output_path, m.group(1) + '.jpg')
      cv2.imwrite(output_filename, img)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, version=__version__)
  parser.add_argument('-l', dest='logLevel', default='info', help="logging level: {debug, info, error}")
  parser.add_argument('--visualize', action='store_true', dest="visualize", help = "visualize")
  parser.add_argument('img_path', help = "supply path to database images")
  parser.add_argument('output_path', help = "supply path to output images")
  args = parser.parse_args()

  logging.basicConfig(level=getattr(logging, args.logLevel.upper()), format='%(asctime)s %(levelname)s %(message)s')
  process(args.img_path, args.output_path, args.visualize)




