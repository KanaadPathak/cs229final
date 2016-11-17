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

#Train file format:
#UCI
#Label, specimen number, eccentricity, ... (14 attributes)
class LeafDataSetUCI():
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

def trainLR(dataSet, CRange=np.logspace(-3,9,13), reTrain=True):
  bestC=1.0; maxScore=0
  for C in CRange:
    clf=linear_model.LogisticRegression(penalty='l2', C=C, solver='sag', multi_class='multinomial', verbose=0, warm_start=False, n_jobs=-1)
    scores = cross_val_score(clf, dataSet.X_train, dataSet.y_train, cv=10); m = scores.mean()
    print("Training Error(C=%f): %0.2f (+/- %0.2f)" % (C, 1-m, scores.std() * 2))
    if m>maxScore: bestC=C; maxScore=m;
  clf=linear_model.LogisticRegression(penalty='l2', C=bestC, solver='sag', multi_class='multinomial', verbose=0, warm_start=False, n_jobs=-1)
  clf.fit(dataSet.X_train, dataSet.y_train)
  return clf

def trainSVMRBF(dataSet, CRange=np.logspace(-4,6,11), gammaRange=np.logspace(-5,9,15)):
  classifiers = []
  maxScore=0
  #TODO: try different types of features
  for C in CRange:
    for gamma in gammaRange:
      clf = SVC(C=C, gamma=gamma)
      scores = cross_val_score(clf, dataSet.X_train, dataSet.y_train, cv=10); m = scores.mean()
      print("Training Error(C=%f gamma=%f): %0.2f (+/- %0.2f)" % (C, gamma, 1-m, scores.std() * 2))
      if m>maxScore: maxScore=m; (bestC, bestGamma) = (C, gamma)
  clf = SVC(C=bestC, gamma=bestGamma)
  clf.fit(dataSet.X_train, dataSet.y_train)
  return clf

def load_kaggle(scale=True):
  return LeafDataSet(os.path.join(os.path.dirname(__file__), '../data/kaggle/train.csv'), scale)

def load_uci(scale=True):
  return LeafDataSetUCI(os.path.join(os.path.dirname(__file__), '../data/uci/leaf.csv'), scale)

def selectModel(dataSet):
  for model in (trainLR, trainSVMRBF):
    clf = algo(dataSet)
    predicts = clf.predict(dataSet.X_test)
    print("%s Test Error: %0.2f" % (str(algo), np.sum(predicts!=dataSet.y_test)/float(np.size(dataSet.y_test))))

def selectFeature(model, dataSet):

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, version=__version__)
  parser.add_argument('trainCsvFile', help = "supply train CSV file")
  args = parser.parse_args()
  dataSet = LeafDataSet(args.trainCsvFile)
  dataSet = load_uci();
  if args.dumpLabel is not None:
    dataSet.writeLabelToJson(args.dumpLabel)
  selectModel(dataSet)
