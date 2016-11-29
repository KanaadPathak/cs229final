#!/usr/bin/env python
"""

"""
__version__ = "0.1"
import argparse, sys, os
import logging
import time
from codebook import *
from image_feature import *
from preprocess import *
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib

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
    joblib.dump(d, filename)

  @classmethod
  def deserialize(clf, filename):
    """ read BoF terms from file
    output: BoF term counts list
    """
    d = joblib.load(filename)
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
    detector = SiftDetector()
    if 'detector' in configs and configs['detector'] is not None:
      detector_type = configs['detector']
      logging.info("using detector: %(detector_type)s" % locals() )
      if detector_type.lower() == 'dense':
        detector = DenseDetector(configs)
      else:
        raise 'unknown detectory type: %(detector_type)s' % locals()
    descriptor = SiftDescriptor(detector)
    self.cb = Codebook.build(configs['train_records_file'], descriptor, ClusterKMeans(), configs['K'])


def process(configs):
  t0= time.time()
  bof = BagOfFeature(configs)
  t1= time.time()
  logging.info("Takes %.2f to build codebook" % (t1-t0))
  for (i,o) in (('train_records_file', 'train_output'), ('test_records_file', 'test_output')):
    t0= time.time()
    data = bof.map_feature(configs[i])
    t1= time.time()
    logging.info("Takes %.2f to assign terms for %s" % (t1-t0, configs[i]) )
    TermSerializer.serialize(configs[o], data)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, version=__version__)
  parser.add_argument('-l', dest='logLevel', default='info', help="logging level: {debug, info, error}")
  for (k, t, h) in (('K', int, "size of the codebook"), ('detector', str, "type of detector to use")):
    parser.add_argument('--'+k, dest=k, type=t, help=h)

  for f in ('train_records_file', 'test_records_file', 'train_output', 'test_output'):
    parser.add_argument(f, help = "supply filename for %s" % f)

  args = parser.parse_args()
  logging.basicConfig(level=getattr(logging, args.logLevel.upper()), format='%(asctime)s %(levelname)s %(message)s')
  logging.debug(str(args))
  process(vars(args))
