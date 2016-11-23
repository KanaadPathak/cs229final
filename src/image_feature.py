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


class HogDescriptor(object):
  """  represents HOG descriptor
  These seems to be fixed in open cv
  #block_size is fixed at (16,16) and cell_size at (8,8)
  #nbins fixed at 9
  The following are configurable but may require regenerate samples
  """
  _parameters = {
    'shrink_w_size' : 128, #shrink before detection
    'window_size' : (128, 256), #detection window size,
    'step_size' : (16,16), # step size of x and y, larger faster with less window, start off from (4,4)
    'padding' : (0,0), # Typical values for padding include (8, 8), (16, 16), (24, 24), and (32, 32)
    'scale' : 1.5, # each level decrease by 1/scale, until image size is less or equal of windows size
    'use_mean_shift' : False, #NMS instead
    'threshold' : 0.2, #threhold of merging two box if they overlap by this percentage, usually 0.3~0.5
  }
  @classmethod
  def get_parameter(cls, k):
    return cls._parameters[k] if k in cls._parameters else None

  def extract(self, img):
    desc = self.descriptor.compute(img)
    return desc.flatten()

  def __init__(self, parameters=None):
    if parameters is not None:
      for (k,v) in parameters.items():
        if k in HogDescriptor._parameters:
          HogDescriptor._parameters[k] = v
    win_size = HogDescriptor._parameters['window_size']
    block_size = (16,16)
    block_stride = (8,8)
    cell_size = (8,8)
    nbins = 9
    self.descriptor = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
