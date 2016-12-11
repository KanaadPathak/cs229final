import os
import shutil
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


def print_cm(cm, classes, top):
    mismatch = []; match = []
    cnt = cm.astype('float').sum(axis=1)
    cnt = np.where(cnt, cnt, np.ones(cnt.shape).astype('float'))
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i != j and cm[i,j] != 0:
            mismatch.append([cm[i,j], i, j, cm.astype('float')[i,j]/cnt[i]])
        if i == j:
            match.append([cm[i,j], i, cm.astype('float')[i,j]/cnt[i]])
    match = sorted(match, reverse = True, key = lambda x: x[0])
    mismatch = sorted(mismatch, reverse = True, key = lambda x: x[0])
    print("top %(top)d matched"%locals())
    for i in range(top):
        item = match[i]
        print('%s(%d): %f(%d)' % (classes[item[1]], item[1], item[2],item[0]))
    print("top %(top)d mismatch, :-("%locals())
    for i in range(top):
        item = mismatch[i]
        print('%s(%d)->%s(%d): %f(%d)' % (classes[item[1]], item[1], classes[item[2]], item[2], item[3], item[0]))

def plot_cm(y_test, y_pred, classes, visual=False, pretty_print=False, output=None, top=10,
                     title='Confusion matrix', cmap=plt.cm.jet):
    """ This function save the confusion matrix """
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    #cm = np.zeros((201,201))

    #for (i,j) in zip(y_test, y_pred):
    #  cm[i,j] += 1
    if pretty_print:
        print_cm(cm, classes, top)

    cnt = cm.sum(axis=1)
    cnt =np.where( cnt, cnt, np.ones(cnt.shape))
    #normalize
    cm = cm.astype('float') / cnt[:, np.newaxis]

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    cax = ax.matshow(cm, cmap=cmap)
    fig.colorbar(cax)
    if visual:
        plt.show()
    if output is not None:
        plt.savefig(output)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('-l', dest='logLevel', default='info', help="logging level: {debug, info, error}")
  parser.add_argument('--output', dest='output', help="save confusion matrix")
  parser.add_argument('--visual', dest='visual', action='store_true', help="save confusion matrix")
  parser.add_argument('--top', dest='top', type=int, default=10, help="save confusion matrix")
  parser.add_argument('--pretty_print', dest='pretty_print', action='store_true', help="print confusion matrix")
  parser.add_argument('result_file', help="results")
  args = parser.parse_args()

  logging.basicConfig(level=getattr(logging, args.logLevel.upper()), format='%(asctime)s %(levelname)s %(message)s')

  data = joblib.load(args.result_file)
  #d = {'y_predict': best_predict, 'y_test': y_test, 'score': best_score,'y_class': test_class}
  y_class = data['y_class']
  y_predict = data['y_predict']
  y_test = data['y_test']
  plot_cm(y_test, y_predict, y_class,
                   visual=args.visual,
                   pretty_print=args.pretty_print,
                   output=args.output,
                   top=args.top,
                   )

