#!/usr/bin/env python
"""

"""
__version__ = "0.2"

import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import offsetbox
from sklearn.preprocessing import StandardScaler
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)
import sys
import argparse
import logging
import bof
from pylab import *
import tables

def plotInputData(dataSet):
  #randomly pick two features and plot it 
  scaler = StandardScaler()
  X = scaler.fit_transform(dataSet.data)

  nrows=3; ncols=4; plot_number=1;
  plt.figure(figsize=(8, 1 * ncols))
  features = (X[:,0:64], X[:,64:128], X[:,128:192])
  names = ("margin", "texture", "shape")
  for (f,s) in zip(features, names):
    for i in range(0, ncols):
      plt.subplot(nrows, ncols, plot_number); plot_number +=1;
      idx= np.random.permutation(64)
      plt.title("%s %d and %d" % (s, idx[0], idx[1]))
      plt.scatter(f[:,idx[0]], f[:,idx[1]], marker='o', c=dataSet.target)
  plt.show()


def get_cmap(N):
    import matplotlib.cm as cmx
    import matplotlib.colors as colors
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, y, N, output):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    cmap = get_cmap(N)
    markers =  ( 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')
    for i in range(X.shape[0]):
        label = y[i]
        plt.scatter(X[i, 0], X[i, 1], color=cmap(label), vmin=1, vmax=N, marker=markers[label%len(markers)],linewidths=0)
    #plt.colorbar()
    plt.show()
    if output is not None:
        plt.savefig(output)


def plot_tsne(X, y, output):
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, random_state=0)
    N = np.max(y)
    logging.info("number of labels: %d"%(N))
    t0 = time.time()
    X_tsne = tsne.fit_transform(X)
    print("t-SNE embedding of the digits (time %.2fs)" % (time.time() - t0))
    plot_embedding(X_tsne, y, N, output)

def load_feature_form_joblib(feature_file):
    from sklearn.externals import joblib
    d = joblib.load(feature_file)
    return d['X'], d['y']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, version=__version__)
    parser.add_argument('-l', dest='logLevel', default='info', help="logging level: {debug, info, error}")
    parser.add_argument('--type', dest='type', default='tv', help="file type {tv, joblib}")
    parser.add_argument('--output', dest='output', help="output file")
    parser.add_argument('data_set', help="supply dataset file")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.logLevel.upper()), format='%(asctime)s %(levelname)s %(message)s')

    if args.type == 'tv':
        data_set = bof.BoFDataSet(args.data_set, args.data_set)
        X = data_set.X_test
        y = data_set.y_test
    else:
        X,y = load_feature_form_joblib(args.data_set)
    logging.info("data set size=%d, feature d=%d"%(len(X), X[0].size))
    plot_tsne(X, y, args.output)
