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
  for i in range(0,len(imgs)):
    if len(imgs[i].shape) == 3:
      converted = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
      plt.subplot(1, len(imgs), 1+i),plt.imshow(converted),plt.title(titles[i])
    else:
      plt.subplot(1, len(imgs), 1+i),plt.imshow(imgs[i], cmap='gray'),plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
  plt.show()


def kmeans_threshold(img, visualize=False):
  #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  orig = img.copy()
  (h,w,d) = img.shape
  img = img.reshape((h * w, d))
  clt = KMeans(2)
  clt.fit(img)

  #green = np.uint8([[[0,255,0]]])
  #green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
  #pos_label = clt.predict(green[0,0])
  neg_label = clt.predict(img[0,:])
  labels = clt.predict(img)

  for i in range(len(labels)):
    #if labels[i]  !=  pos_label:
    if labels[i]  ==  neg_label:
      img[i] = [0, 0, 0]

  img = img.reshape((h, w, d))
  #img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
  if visualize:
    showImages((orig, img), ("origin", "kmeans"))
  return img

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
  logging.debug("applying morphology")
  #se = np.ones((2, 2), dtype='uint8')
  #canny = cv2.morphologyEx(canny, cv2.MORPH_OPEN, se)

  se = np.ones((7, 7), dtype='uint8')
  canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, se)
  canny = cv2.dilate(canny, se )


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

  #init mask for grabCut
  mask = np.ones(img_contour.shape[:2], np.uint8) * cv2.GC_BGD
  #pick = max(int(len(perimeter) * 0.1 + 1), min(len(perimeter), 3))
  pick = min(len(perimeter), 3)
  logging.debug(" found contours: %d use: %d", len(perimeter), pick)
  for i in range(0, pick):
    index = perimeter[i][1]
    max_index.append(index)
    cv2.drawContours(img_contour, contours, index, (255,0,0), 3)

  # Get convex hull for max contours and draw them
  cont = np.vstack(contours[i] for i in max_index)
  hull = cv2.convexHull(cont)
  unified.append(hull)
  cv2.drawContours(img_contour,unified,-1,(0,0,255),3)

  return img_contour, contours, perimeter, hull, max_index


'''
Given a convex hull apply graph cut to the image
Assumptions:
- Everything inside convex hull is the foreground object - cv2.GC_FGD or 1
- Everything outside the rectangle is the background -  cv2.GC_BGD or 0
- Between the hull and the rectangle is probably foreground - cv2.GC_PR_FGD or 3
'''
def cut_graph_from_hull(hull,img, contours, max_index):
  # First create our rectangle that contains the object
  y_corners = np.amax(hull, axis=0)
  x_corners = np.amin(hull,axis=0)
  x_min = x_corners[0][0]
  x_max = x_corners[0][1]
  y_min = y_corners[0][0]
  y_max = y_corners[0][1]

  # input/output mask
  mask = np.ones(img.shape[:2],np.uint8) * cv2.GC_BGD

  # Values needed for algorithm
  bgdModel = np.zeros((1,65),np.float64)
  fgdModel = np.zeros((1,65),np.float64)

  #method #1 use GC_INIT_WITH_RECT
  #rect = (x_min,x_max,y_min,y_max)
  #cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

  #method #2 use GC_INIT_WITH_MASK
  rect = None
  hull_lines = [hull]
  #everything inside the hull is probabably
  #cv2.drawContours(mask, hull_lines,-1, (cv2.GC_PR_FGD,cv2.GC_PR_FGD,cv2.GC_PR_FGD) ,-1)
  #mark everything inside the contours are true
  for i in max_index:
    cv2.drawContours(mask, contours, i, (cv2.GC_FGD, cv2.GC_FGD, cv2.GC_FGD), -1)

  cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

  #apply mask to the image
  mask2 = np.where((mask==cv2.GC_PR_BGD)|(mask==cv2.GC_BGD),0,1).astype('uint8')
  img = img*mask2[:,:,np.newaxis]
  return (img, mask)

def auto_canny(image, sigma=0.80):
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

def grabcut_from_contour(orig, visualize=False):
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
  img_contour, contours, perimeters, hull, max_index = largestContours(canny,img,img_gray, visualize)
  if img_contour is None:
    logging.error("hitting error, return original image")
    return orig

  # Grabcut - Same bounding box than contours...
  (img_grcut, mask) = cut_graph_from_hull(hull, img, contours, max_index)

  ## Show images
  images = [img, canny, img_contour, mask, img_grcut ]
  titles = ["original", "edges", "largest contour", "mask", "graph cut"]
  if visualize:
    showImages(images, titles)
  return (img_grcut, img_contour)


def normalize(img):
  #http://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c/24341809#24341809
  lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  # -----Splitting the LAB image to different channels-------------------------
  l, a, b = cv2.split(lab)

  # -----Applying CLAHE to L-channel-------------------------------------------
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  cl = clahe.apply(l)

  limg = cv2.merge((cl, a, b))

  # -----Converting image from LAB Color model to RGB model--------------------
  return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)



def process(args):
  img_path = args['img_path']
  output_root = args['output_path']
  visualize = args['visualize']
  method = args['method']
  save_debug = args['save_debug']
  shrink = args['shrink']
  w = args['shrink_w']
  h = args['shrink_h']
  debug_images = None
  for subdir, dirs, files in os.walk(img_path):
    for f in files:
      m = re.match(r"(.*)[.]jpg", f)
      if m is None:
        continue
      filename = os.path.join(subdir,f)
      logging.debug("working on %(filename)s" % locals())
      img = cv2.imread(filename)
      img = normalize(img)
      if method == 'shape':
        (img, debug_images) = grabcut_from_contour(img, visualize)
      elif method == 'color':
        img = kmeans_threshold(img, visualize)
      elif method == 'nop':
        pass
      else:
        raise ValueError('unknown method')
      if shrink:
        img = imgutil.shrink(img, w, h)
      output_path = os.path.join(output_root, os.path.basename(subdir))
      if not os.path.exists(output_path):
        os.mkdir(output_path)
      output_filename = os.path.join(output_path, m.group(1) + '.jpg')
      assert cv2.imwrite(output_filename, img)
      logging.debug("writing to %(output_filename)s" % locals())
      if save_debug and debug_images is not None:
        output_filename = os.path.join(output_path, m.group(1) + '.debug.jpg')
        assert cv2.imwrite(output_filename, debug_images)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, version=__version__)
  parser.add_argument('-l', dest='logLevel', default='info', help="logging level: {debug, info, error}")
  parser.add_argument('--method', dest='method', default="nop",
                      help = "method to remove background {shape, color, nop}")
  parser.add_argument('--visualize', action='store_true', dest="visualize", default=False, help = "visualize")
  parser.add_argument('--save_debug', action='store_true', dest="save_debug", default=False, help = "visualize")
  parser.add_argument('--shrink', action='store_true', dest="shrink", default=True, help = "shrink to (256,256)")
  parser.add_argument('--shrink_w', dest="shrink_w", type=int, default=256, help = "shrink width ")
  parser.add_argument('--shrink_h', dest="shrink_h", type=int, default=256, help = "shrink height ")
  parser.add_argument('img_path', help = "supply path to database images")
  parser.add_argument('output_path', help = "supply path to output images")
  args = parser.parse_args()

  logging.basicConfig(level=getattr(logging, args.logLevel.upper()), format='%(asctime)s %(levelname)s %(message)s')
  process(vars(args))




