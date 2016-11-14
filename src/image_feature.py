#!/usr/bin/env python
"""

"""
__version__ = "0.1"
import cv2
import logging
import numpy as np
import preprocess
from abc import ABCMeta, abstractmethod

class ImageKeypoint(object):
  """ Represents image key points """
  __metaclass__ = ABCMeta
  @abstractmethod
  def detect(self, img): pass


class SiftDetector(ImageKeypoint):
  def detect(self, img):
    sift = cv2.xfeatures2d.SIFT_create()
    return sift.detect(img, None) 


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

