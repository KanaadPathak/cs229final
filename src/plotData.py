#!/usr/bin/env python
"""

"""
__version__ = "0.1"
import numpy as np
import matplotlib.pyplot as plt
from leafClassifier import *

def plotData(dataSet):
  #randomly pick two features and plot it 
  nrows=3; ncols=8; plot_number=1;
  plt.figure(figsize=(8, 1 * ncols))
  features = (10 * dataSet.margin, 10 * dataSet.texture, 200 * dataSet.shape)
  names = ("margin", "texture", "shape")
  for (f,s) in zip(features, names):
    for i in range(0, ncols):
      plt.subplot(nrows, ncols, plot_number); plot_number +=1;
      idx= np.random.permutation(64)
      plt.title("%s %d and %d" % (s, idx[0], idx[1]))
      plt.scatter(f[:,idx[0]], f[:,idx[1]], marker='o', c=dataSet.target)
  plt.show()

if __name__ == "__main__":
  dataSet = load_kaggle()
  plotData(dataSet)
