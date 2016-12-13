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
def plot_embedding(X, y, N):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = get_cmap(N)
    for i in range(X.shape[0]):
        ax1.scatter(X[i, 0], X[i, 1], color=cmap(y[i]))

        #plt.xticks([]), plt.yticks([])
        #if title is not None:
        #plt.title(title)
    plt.show()

def plot_tsne(X, y):
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, random_state=0)
    N = np.max(y) + 1
    t0 = time.time()
    X_tsne = tsne.fit_transform(X)
    print("t-SNE embedding of the digits (time %.2fs)" % (time.time() - t0))
    plot_embedding(X_tsne, y, N)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, version=__version__)
    parser.add_argument('-l', dest='logLevel', default='info', help="logging level: {debug, info, error}")
    parser.add_argument('train_file', help = "supply train file")
    parser.add_argument('test_file', help = "supply test file")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.logLevel.upper()), format='%(asctime)s %(levelname)s %(message)s')

    data_set = bof.BoFDataSet(args.train_file, args.test_file)
    plot_tsne(data_set.X_test, data_set.y_test)
