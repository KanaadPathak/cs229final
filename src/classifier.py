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
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression, RFECV
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
from sklearn.decomposition import PCA

class Classifier(object):
  def score(self, data_set):
    return self.clf.score(data_set.X_test, data_set.y_test)

  def predict(self, data_set):
    return self.clf.predict(data_set.X_test)

class SoftMax(Classifier):
  def __init__(self, data_set, configs):
    parameters = {'C': configs['c_range']}
    clf=linear_model.LogisticRegression(penalty='l2', solver='newton-cg', multi_class='multinomial')
    self.clf = GridSearchCV(clf, parameters, n_jobs=1)
    self.clf.fit(data_set.X_train, data_set.y_train)
    logging.info("Training Accuracy(C=%f) %.4f" % (self.clf.best_params_['C'], self.clf.best_score_))

class SVM(Classifier):
  def __init__(self, data_set, parameters):
    svr = SVC()
    verbose = 0
    if 'verbose' in parameters:
      verbose = parameters['verbose']
    self.clf = GridSearchCV(svr, parameters, verbose=verbose, n_jobs=-1)
    self.clf.fit(data_set.X_train, data_set.y_train)

class SVMGaussianKernel(SVM):
  def __init__(self, data_set, configs):
    parameters = {'kernel': ['rbf'], 'C': configs['c_range'], 'gamma': configs['gamma_range']}
    SVM.__init__(self, data_set, parameters)
    s = ""
    for k in ('C', 'gamma'):
      s += " {0:s}={1:.2f} ".format(k, self.clf.best_params_[k])
    logging.info("Training Accuracy(%s): %.4f" % (s, self.clf.best_score_))

class SVMLinearKernel(SVM):
  def __init__(self, data_set, configs):
    parameters = {'kernel': ['linear'], 'C': configs['c_range']}
    SVM.__init__(self, data_set, parameters)
    logging.info("Training Accuracy(C=%f) %.4f" % (self.clf.best_params_['C'], self.clf.best_score_))

def selectModel(data_set, models):
  scaler = StandardScaler()
  data_set.X_train = scaler.fit_transform(data_set.X_train)
  data_set.X_test = scaler.transform(data_set.X_test)
  #print("after scaling")
  #print(data_set.X_train.shape)
  # selector1= VarianceThreshold(threshold=(.9 * (1 - .9)))
  # X_train = selector1.fit_transform(X_train, y_train)
  # selector2 = SelectKBest(f_classif, k=min(X_train.shape[1], 3000))
  # X_train = selector2.fit_transform(X_train, y_train)

  # dimension reduction
  #pca = PCA()
  #data_set.X_train = pca.fit_transform(data_set.X_train)
  #print("after PCA")
  #print(data_set.X_train.shape)

  # feature selection
  # backward search
  #svc = SVC(kernel="linear", C=0.001)
  #rfecv = RFECV(estimator=svc, step=10, cv=StratifiedKFold(3), n_jobs=-1, scoring='accuracy', verbose=9)
  #data_set.X_train = rfecv.fit_transform(data_set.X_train, data_set.y_train)
  #print("Backward search gives number of features : %d" % rfecv.n_features_)

  #data_set.X_test = scaler.transform(data_set.X_test)
  #data_set.X_test = rfecv.predict(data_set.X_test)

  for (model, configs) in models:
    if 'skip' in configs and configs['skip']: continue
    t0 = time.time()
    clf = model(data_set, configs)
    score = clf.score(data_set)
    logging.info("%s (%.2f) Test Accuracy: %0.4f" % (str(model), time.time()-t0, score))

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
     'c_range' : np.logspace(2, 6, 2), #np.logspace(1, 9, 6, endpoint=True),
      #gamma = 1/(2*tao^2)
     'gamma_range' : np.logspace(-3, 2, 5), #np.linspace(1.0/(2*8*8), 1, 1, endpoint=True),
     'skip' : False,
    },
    {
      #SVMGaussianLinear
      'c_range' : np.logspace(-1, 5, 6, endpoint=True), #np.logspace(-4, 6, 11),
      'skip' : False,
    },
    {
      #Softmax
      'c_range' : np.logspace(-4, 6, 11, endpoint=True), #np.logspace(-4, 6, 11)
      'skip' : True,
    },
  )
  selectModel(data_set, zip(classifiers, configs))
