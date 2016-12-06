import os
import shutil
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

def plot_confusion_matrix(y_test, y_pred,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    #cm = confusion_matrix(y_test, y_pred)
    cm = np.zeros((201,201))

    for (i,j) in zip(y_test, y_pred):
      cm[i,j] += 1
    np.set_printoptions(precision=2)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    mismatch = []
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        if i != j and cm[i,j] != 0:
          #print('%d->%d: %d' %( i, j, cm[i,j]))
          mismatch.append([cm[i,j], i, j])
    mismatch = sorted(mismatch, reverse = True, key = lambda x: x[0])
    for i in range(50):
      item = mismatch[i]
      print('%s(%d)->%s(%d): %d' % (classes[item[1]], item[1], classes[item[2]], item[2], item[0]))


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('-l', dest='logLevel', default='info',
                      help="logging level: {debug, info, error}")
  parser.add_argument('result_file', help="results")
  args = parser.parse_args()

  logging.basicConfig(level=getattr(logging, args.logLevel.upper()),
                      format='%(asctime)s %(levelname)s %(message)s')

  data = joblib.load(args.result_file)
  #d = {'y_predict': best_predict, 'y_test': y_test, 'score': best_score,'y_class': test_class}
  y_class = data['y_class']
  y_predict = data['y_predict']
  y_test = data['y_test']
  score = data['score']
  plot_confusion_matrix(y_test, y_predict, y_class)
  logging.info("done")

