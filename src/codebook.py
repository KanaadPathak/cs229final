#!/usr/bin/env python
"""

"""
__version__ = "0.1"
import argparse, sys, os
import cv2
import logging
import numpy as np
from abc import ABCMeta, abstractmethod
from image_feature import *
from preprocess import *
from sklearn.cluster import KMeans

class ClusterKMeans(object):
  @classmethod
  def fit(cls, K, vectors):
    #TODO
    estimator = KMeans(n_clusters=K, random_state=0, n_init=1, verbose=1).fit(vectors)
    return estimator


class Codebook(object):
  """ Represents visual codebook of BoF approach """
  @classmethod
  def build(cls, filename, descriptor, cluster, K):
    vectors = np.zeros((0, descriptor.get_size()))
    records = ImageRecordSerializer.deserialize(filename)
    #TODO: For performance, we use parts of images for building the codebook
    for (i,r) in enumerate(records[:100]):
      vectors = np.concatenate((vectors,descriptor.extract(r[2])))
    #TODO: PCA
    estimator = ClusterKMeans.fit(K, vectors)
    logging.debug("distortion function: %f"%estimator.inertia_)
    logging.debug("cluster size histogram:\n %s" % (str(np.bincount(estimator.labels_))))
    return Codebook(estimator, descriptor)

  def __init__(self, estimator, descriptor):
    self.estimator = estimator
    self.descriptor = descriptor

  def assign_term(self, img):
    fv = self.descriptor.extract(img)
    assert fv is not None, 'image descriptor is empty'
    return self.estimator.predict(fv)


  def get_clustersize(self):
    return self.estimator.n_clusters

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, version=__version__)
  parser.add_argument('-l', dest='logLevel', default='info',
      help="logging level: {debug, info, error}")
  parser.add_argument('train_file', help = "supply filename for preprocessed training data")
  args = parser.parse_args()

  logging.basicConfig(level=getattr(logging, args.logLevel.upper()),
                      format='%(asctime)s %(levelname)s %(message)s')

