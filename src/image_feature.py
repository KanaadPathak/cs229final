#!/usr/bin/env python
"""

"""
__version__ = "0.1"
import cv2
import logging
import numpy as np
import preprocess
from abc import ABCMeta, abstractmethod

class ImageDetector(object):
  """ Represents image key points detector"""
  def detect(self, img):
    return self.detector.detect(img, None)

  def __init__(self):
    self.detector = None

class SiftDetector(ImageDetector):
  def __init__(self):
    ImageDetector.__init__(self)
    self.detector = cv2.xfeatures2d.SIFT_create()


class DenseDetector(ImageDetector):
  def detect(self, img):
    kp = [cv2.KeyPoint(x, y, self.configs['step_size']) for y in range(0, img.shape[0], self.configs['step_size'])
          for x in range(0, img.shape[1], self.configs['step_size'])]
    return kp

  def __init__(self, configs):
    ImageDetector.__init__(self)
    self.configs = {
      'step_size' : 20,
    }
    for (k,v) in self.configs.items():
      if configs is not None and k in configs:
        self.configs[k] = configs[k]


class ImageFeatureDescriptor(object):
  """ Represents image feature vectors """
  __metaclass__ = ABCMeta
  @abstractmethod
  def extract(self, img): pass

  def get_size(self):
    """ return size of descriptor"""
    pass

  def __init__(self, detector):
    self.detector = detector

class SiftDescriptor(ImageFeatureDescriptor):
  def extract(self, img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = self.detector.detect(img)
    (kp, desc) = sift.compute(img, kp, None)
    return desc

  def get_size(self):
    return 128

