#!/usr/bin/env python
"""

"""
__version__ = "0.1"
import argparse, sys, os
import json
import numpy as np
from collections import OrderedDict
from sklearn import linear_model, datasets
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#Train file format:
#1st row -- comments
#subsequent rows:
#id,species(label),margin1,...,margin64,shape1,...,shape64,texture1,...,texture64

class LeafDataSet():
  def __init__(self, csvfilename):
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

  def writeLabelToJson(self, filename):
    orderLabels = OrderedDict(sorted(self.labels.items(), key= lambda t: t[1]))
    with open(filename, 'w') as fd:
      json.dump(orderLabels,fd) 

def trainLR(dataSet):
  bestC=1.0; minScore=float("inf")
  #<0.1 gives ~0.56 accuracy
  for C in np.linspace(0.01,0.1,10):
    clf=linear_model.LogisticRegression(penalty='l2', C=C, solver='sag', multi_class='multinomial', verbose=0, warm_start=False, n_jobs=-1)
    scores = cross_val_score(clf, dataSet.data, dataSet.target, cv=10); m = scores.mean()
    print("Accuracy(C=%f): %0.2f (+/- %0.2f)" % (C, m, scores.std() * 2))
    if m<minScore: bestC=C; minScore=m;
  return (clf,bestC)

def trainSVMRBF(dataSet):
  #usually a good idea to scale data for SVM training
  scaler = StandardScaler()
  X = scaler.fit_transform(dataSet.data)
  C_2d_range = np.logspace(-2, 10, 13)
  gamma_2d_range = np.logspace(-1, 3, 5)
  classifiers = []
  minScore=float("inf");
  #TODO: try different types of features
  for C in C_2d_range:
    for gamma in gamma_2d_range:
      clf = SVC(C=C, gamma=gamma)
      scores = cross_val_score(clf, X, dataSet.target, cv=10); m = scores.mean()
      print("Accuracy(C=%f gamma=%f): %0.2f (+/- %0.2f)" % (C, gamma, m, scores.std() * 2))
      if m<minScore: minScore=m; (bestC, bestGamma) = (C, gamma)
  return (clf, bestC, bestGamma)

def load_kaggle():
  return LeafDataSet(os.path.join(os.path.dirname(__file__), '../data/kaggle/train.csv'))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, version=__version__)
  parser.add_argument('trainCsvFile', help = "supply train CSV file")
  parser.add_argument('--dumpLabel', help = "dump labels to plist")
  args = parser.parse_args()
  dataSet = LeafDataSet(args.trainCsvFile)
  if args.dumpLabel is not None:
    dataSet.writeLabelToJson(args.dumpLabel)
  
