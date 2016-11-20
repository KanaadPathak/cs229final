#!/usr/bin/env python
"""

"""
__version__ = "0.2"
import argparse, sys, os
import bof
import dataset
import json
import logging
import numpy as np
import time

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from sklearn import linear_model, datasets, svm
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class Classifier(object):
  def score(self, data_set):
    return self.clf.score(data_set.X_test, data_set.y_test)

  def predict(self, data_set):
    return self.clf.predict(data_set.X_test)

class SoftMax(Classifier):
  def __init__(self, data_set, configs):
    best_c=1.0; max_score=0
    for c in configs['c_range']:
      best_c = 0.1
      clf=linear_model.LogisticRegression(penalty='l2', C=c, solver='sag', multi_class='multinomial', verbose=0, warm_start=False, n_jobs=-1)
      scores = cross_val_score(clf, data_set.X_train, data_set.y_train, cv=10);
      m = scores.mean()
      logging.debug("Training Accuracy (c=%f): %0.2f (+/- %0.2f)" % (c, m, scores.std() * 2))
      if m>max_score: best_c=c; max_score=m;
    clf=linear_model.LogisticRegression(penalty='l2', C=best_c, solver='sag', multi_class='multinomial', verbose=0, warm_start=False, n_jobs=-1)
    clf.fit(data_set.X_train, data_set.y_train)
    logging.info("Training Accuracy(C=%f): %.2f" % (best_c, max_score))
    self.clf = clf

class SVM(Classifier):
  def __init__(self, data_set, parameters):
    svr = SVC()
    self.clf = GridSearchCV(svr, parameters)
    self.clf.fit(data_set.X_train, data_set.y_train)

class SVMGaussianKernel(SVM):
  def __init__(self, data_set, configs):
    parameters = {'kernel': ['rbf'], 'C': configs['c_range'], 'gamma': configs['gamma_range']}
    SVM.__init__(self, data_set, parameters)
    s = ""
    for k in ('C', 'gamma'):
      s += " {0:s}={1:.2f} ".format(k, self.clf.best_params_[k])
    logging.info("Training Accuracy(%s): %.2f" % (s, self.clf.best_score_))

class SVMLinearKernel(SVM):
  def __init__(self, data_set, configs):
    parameters = {'kernel': ['linear'], 'C': configs['c_range']}
    SVM.__init__(self, data_set, parameters)
    logging.info("Training Accuracy(C=%f) %.2f" % (self.clf.best_params_['C'], self.clf.best_score_))

def selectModel(data_set, models):
  for (model, configs) in models:
    if 'skip' in configs and configs['skip']: continue
    t0 = time.time()
    clf = model(data_set, configs)
    score = clf.score(data_set)
    logging.info("%s (%.2f) Test Accuracy: %0.2f" % (str(model), time.time()-t0, score))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, version=__version__)
  parser.add_argument('-l', dest='logLevel', default='info',
                      help="logging level: {debug, info, error}")
  parser.add_argument('train_file', help = "supply train file")
  parser.add_argument('test_file', help = "supply test file")
  args = parser.parse_args()

  logging.basicConfig(level=getattr(logging, args.logLevel.upper()),
                      format='%(asctime)s %(levelname)s %(message)s')

  data_set = bof.BoFDataSet(args.train_file, args.test_file)
  #data_set = dataset.PSDataSet(args.train_file, args.test_file)
  logging.info("Train data: %d x %d"%(len(data_set.X_train), data_set.X_train[0].size))
  logging.info("Test data: %d x %d"%(len(data_set.X_test), data_set.X_test[0].size))

  classifiers = (SVMGaussianKernel, SVMLinearKernel, SoftMax)
  configs=(
    {
      #SVMGaussianKernel
     'c_range' : np.logspace(-4, 6, 11), #np.logspace(1, 9, 6, endpoint=True),
      #gamma = 1/(2*tao^2)
     'gamma_range' : np.logspace(-5, 9, 15), #np.linspace(1.0/(2*8*8), 1, 1, endpoint=True),
     'skip' : False,
    },
    {
      #SVMGaussianLinear
      'c_range' : np.logspace(-4, 6, 11, endpoint=True), #np.logspace(-4, 6, 11),
      'skip' : False,
    },
    {
      #Softmax
      'c_range' : np.logspace(-4, 6, 11, endpoint=True), #np.logspace(-4, 6, 11)
      'skip' : False,
    },
  )
  selectModel(data_set, zip(classifiers, configs))