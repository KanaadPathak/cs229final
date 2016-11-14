#!/usr/bin/env python
"""

"""
__version__ = "0.1"
import argparse, sys, os
import cv2
import logging
import numpy as np
import pickle
from abc import ABCMeta, abstractmethod
from codebook import *
from image_feature import *
from preprocess import *
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse

class TermSerializer(object):
  @classmethod
  def serialize(clf, filename, data):
    """ write BoF terms into to file
    """
    (labels, term_vector) = zip(*data)
    assert (len(labels) == len(term_vector))
    d = {
      'labels': labels,
      'term_vector': term_vector,
    }
    with open(filename, 'w') as fd:
      pickle.dump(d, fd)

  @classmethod
  def deserialize(clf, filename):
    """ read BoF terms from file
    output: BoF term counts list
    """
    with open(filename, 'r') as fd:
      d = pickle.load(fd)
    return zip(d['labels'], d['term_vector'])

class BoFDataSet(object):
  def __init__(self, train_file, test_file):
    (self.y_train, self.X_train) = zip(*TermSerializer.deserialize(train_file))
    (self.y_test, self.X_test) = zip(*TermSerializer.deserialize(test_file))

class BagOfFeature(object):
  def map_feature(self, filename):
    records = ImageRecordSerializer.deserialize(filename)
    K = self.cb.get_clustersize()
    m=len(records)
    #terms count per data: m x K
    counts = np.zeros((m, K), dtype=float)
    countsPerCode = np.zeros((m, K), dtype=float)
    for (i,r) in enumerate(records):
      logging.debug("%d: mapping %s "%(i, r[1]))
      #count the number of times code j is present in image i
      for j in self.cb.assign_term(r[2]):
        counts[i,j] += 1
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(counts)
    (labels, names, images) = zip(*records)
    return zip(labels, tfidf.toarray())

  def __init__(self, configs):
    descriptor = SiftDescriptor(SiftDetector())
    self.cb = Codebook.build(configs['train_records_file'], descriptor, ClusterKMeans(), configs['K'])


def process(configs):
  bof = BagOfFeature(configs)
  for (i,o) in (('train_records_file', 'train_output'), ('test_records_file', 'test_output')):
    data = bof.map_feature(configs[i])
    TermSerializer.serialize(configs[o], data)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, version=__version__)
  parser.add_argument('-l', dest='logLevel', default='info',
      help="logging level: {debug, info, error}")

  for (k,h) in (('K', "size of the codebook"),):
    parser.add_argument('--'+k, dest=k, type=int, help=h)

  for f in ('train_records_file', 'test_records_file', 'train_output', 'test_output'):
    parser.add_argument(f, help = "supply filename for %s" % f)

  args = parser.parse_args()
  logging.basicConfig(level=getattr(logging, args.logLevel.upper()),
                      format='%(asctime)s %(levelname)s %(message)s')
  logging.debug(str(args))
  process(vars(args))
