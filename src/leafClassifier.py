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
#1st row -- comments
#subsequent rows:
#id,species(label),margin1,...,margin64,shape1,...,shape64,texture1,...,texture64

class LeafDataSet():
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

def trainLR(dataSet, reTrain=True):
  bestC=1.0; minScore=float("inf")
  if not reTrain:
  #<0.1 gives accuracy
    bestC = 0.1
  else:
    for C in np.linspace(0.01,0.1,10):
      clf=linear_model.LogisticRegression(penalty='l2', C=C, solver='sag', multi_class='multinomial', verbose=0, warm_start=False, n_jobs=-1)
      scores = cross_val_score(clf, dataSet.X_train, dataSet.y_train, cv=10); m = scores.mean()
      print("Training Error(C=%f): %0.2f (+/- %0.2f)" % (C, 1-m, scores.std() * 2))
      if m<minScore: bestC=C; minScore=m;
  clf=linear_model.LogisticRegression(penalty='l2', C=bestC, solver='sag', multi_class='multinomial', verbose=0, warm_start=False, n_jobs=-1)
  clf.fit(dataSet.X_train)
  return (clf,bestC)

def trainSVMRBF(dataSet, reTrain=True):
  if not reTrain:
    #gamma=5 give 0.97. very sensitive to gamma, not C
    bestC = 0.1; bestGamma = 5;
  else:
    C_2d_range = np.logspace(-1, 1, 3)
    gamma_2d_range = np.linspace(1, 10, 20)
    classifiers = []
    minScore=float("inf");
    #TODO: try different types of features
    for C in C_2d_range:
      for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        scores = cross_val_score(clf, dataSet.X_train, dataSet.y_train, cv=10); m = scores.mean()
        print("Training Error(C=%f gamma=%f): %0.2f (+/- %0.2f)" % (C, gamma, 1-m, scores.std() * 2))
        if m<minScore: minScore=m; (bestC, bestGamma) = (C, gamma)
  clf = SVC(C=bestC, gamma=bestGamma)
  clf.fit(dataSet.X_train)
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
  for algo in (trainLR, trainSVMRBF):
    clf = algo(dataSet)
    predicts = clf.predict(dataSet.X_test)
    print("%s Test Error: %0.2f" % (str(algo), np.sum(predicts!=dataSet.y_test)/np.size(dataSet.y_test.shape)))
