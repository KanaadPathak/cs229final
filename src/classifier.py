#!/usr/bin/env python
"""

"""
__version__ = "0.2"
import argparse, sys, os
import bof
import json
import numpy as np
import time

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from sklearn import linear_model, datasets
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, train_test_split
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
      print("Training Error(c=%f): %0.2f (+/- %0.2f)" % (c, 1-m, scores.std() * 2))
      if m>max_score: best_c=c; max_score=m;
    clf=linear_model.LogisticRegression(penalty='l2', C=best_c, solver='sag', multi_class='multinomial', verbose=0, warm_start=False, n_jobs=-1)
    clf.fit(data_set.X_train, data_set.y_train)
    self.clf = clf

class SVMRBF(Classifier):
  def __init__(self, data_set, configs):
    max_score=0
    for c in configs['c_range']:
      for gamma in configs['gamma_range']:
        clf = SVC(C=c, gamma=gamma)
        scores = cross_val_score(clf, data_set.X_train, data_set.y_train, cv=10);
        m = scores.mean()
        print("Training Error(C=%f gamma=%f): %0.2f (+/- %0.2f)" % (c, gamma, 1-m, scores.std() * 2))
        if m>max_score:
          max_score=m; (best_c, best_g) = (c, gamma)
    self.clf = SVC(C=best_c, gamma=best_g)
    self.clf.fit(data_set.X_train, data_set.y_train)

def selectModel(data_set, models):
  for (model, configs) in models:
    t0 = time.time()
    clf = model(data_set, configs)
    score = clf.score(data_set)
    print("%s (%.2f) Test Accurary: %0.2f" % (str(model), time.time()-t0, score))
    #predicts = clf.predict(data_set)
    #print predicts

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, version=__version__)
  parser.add_argument('train_file', help = "supply train file")
  parser.add_argument('test_file', help = "supply test file")
  args = parser.parse_args()
  data_set = bof.BoFDataSet(args.train_file, args.test_file)
  classifiers = (SoftMax, SVMRBF)
  configs=(
    {
     'c_range' : np.linspace(20, 20, 1, endpoint=True) #np.logspace(-4, 6, 11)
    },
    {
     'c_range' : np.linspace(100.0, 101.0, 1, endpoint=True),  #np.logspace(-4, 6, 11),
     'gamma_range' : np.linspace(0.1, 2.0, 5, endpoint=True) #np.logspace(-5, 9, 15),
    }
  )
  selectModel(data_set, zip(classifiers, configs))
