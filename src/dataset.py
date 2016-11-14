#!/usr/bin/env python
"""

"""
__version__ = "0.1"
import argparse, sys, os
import json
import numpy as np
from collections import OrderedDict
from sklearn import linear_model, datasets
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#Train file format:
#Kaggle
#1st row -- comments
#subsequent rows:
#id,species(label),margin1,...,margin64,shape1,...,shape64,texture1,...,texture64
class KaggleCsvDataSet(object):
  def __init__(self, csvfilename, scale=True):
    #read in features
    self.data = np.loadtxt(csvfilename, delimiter=',', usecols=range(2,194), skiprows=1)
    self.margin = self.data[:,0:64]
    self.shape = self.data[:,64:128]
    self.texture = self.data[:,128:192]
    #then read in labels sperately
    with open(csvfilename, 'r') as fd:
      #skip the header
      fd.readline()
      self.labels={}; idx=0; y=[]
      for line in fd:
        label = line.split(',')[1]
        if label not in self.labels:
          self.labels[label] = idx; idx += 1
        y.append(self.labels[label])
    self.target = np.fromiter(iter(y), dtype=np.int)
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.target, test_size=0.3)
    if scale:
      scaler = StandardScaler()
      self.X_train = scaler.fit_transform(self.X_train)
      self.X_test = scaler.transform(self.X_test)

  def writeLabelToJson(self, filename):
    orderLabels = OrderedDict(sorted(self.labels.items(), key= lambda t: t[1]))
    with open(filename, 'w') as fd:
      json.dump(orderLabels,fd)

    #Train file format:
#UCI
#Label, specimen number, eccentricity, ... (14 attributes)
class UCICsvDataSet(object):
  def __init__(self, csvfilename, scale=True):
    #read in features
    self.data = np.loadtxt(csvfilename, delimiter=',', usecols=range(2,16))
    #then read in labels sperately
    self.target = np.loadtxt(csvfilename, delimiter=',', usecols=range(0,1), dtype=np.int)
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.target, test_size=0.3)
    if scale:
      scaler = StandardScaler()
      self.X_train = scaler.fit_transform(self.X_train)
      self.X_test = scaler.transform(self.X_test)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, version=__version__)
  parser.add_argument('trainCsvFile', help = "supply train CSV file")
  args = parser.parse_args()
  data_set = KaggleCsvDataSet(args.trainCsvFile)
  if args.dumpLabel is not None:
    data_set.writeLabelToJson(args.dumpLabel)
