#!/usr/bin/env python
"""

"""
__version__ = "0.1"
import argparse, sys, os
import cv2
import classifier
import image_feature
import imgutil
import imutils
import logging
import math
import preprocess
import nms
import numpy as np
import re
import time
import threshold
import json
from sklearn.externals import joblib

class HogDataSet(object):
  def deserialize(self, filename):
    records  =  preprocess.ImageRecordSerializer.deserialize(filename)
    (labels, names, vectors) = zip(*records)
    return (labels, vectors)

  def __init__(self, train_file):
    self.X_test = []; self.y_test = []
    self.X_train = []; self.y_train = []
    if train_file is not None:
      (self.y_train, self.X_train) = self.deserialize(train_file)

class HogDetector(object):
  def __init__(self, parameters):
    self.model_file = parameters['model_file']
    if parameters['action']=='detect' and os.path.exists(self.model_file):
      #deserialize trained model
      logging.info("found model in %s" % (self.model_file))
      self.deserialize()
    else:
      #otherwise starting training
      logging.info("training classifier")
      train_file = parameters['train_file']
      assert os.path.isfile(train_file)
      configs = {
        'c_range' : np.logspace(-2, 2, 5, endpoint=True), #np.logspace(-4, 6, 11, endpoint=True),
        'gamma_range' : np.logspace(-5, 9, 15), #np.linspace(1.0/(2*8*8), 1, 1, endpoint=True),
      }
      t0 = time.time()
      self.clf = classifier.SVMLinearKernel(HogDataSet(train_file), configs).clf
      logging.info("takes %.2f seconds" % (time.time()-t0))
      self.serialize()
    self.hog = image_feature.HogDescriptor()

  def serialize(self):
    joblib.dump(self.clf, self.model_file)

  def deserialize(self):
    self.clf = joblib.load(self.model_file)

  def write_detected_img(self, img, detected, image_name, output_path, p):
    once = p['once']
    visual = p['visual_detection']
    if once:
      detected = detected[:1]
    for (i, (x,y,m,w,h)) in enumerate(detected):
      suffix = ''
      if not once:
        suffix =  '_'  + str(int(m*100)) + '_' + str(i)
      filename = os.path.join(output_path, os.path.basename(image_name).split('.')[0] + suffix + '.jpg')
      if y + h > img.shape[0]:
        y = 0; h = img.shape[0]
      elif x + w > img.shape[1]:
        x = 0; w = img.shape[0]
      logging.debug(" writing detection: margin=%(m).2f (%(x)d, %(y)d, %(w)d, %(h)d)" % locals())
      #apply background removal
      crop = img[y:y + h, x:x + w]
      output = threshold.remove_background(crop, visual)
      cv2.imwrite(filename, output)

  def show_window(self, img, (x,y), window_size, color=(255,255,255)):
    clone = img.copy()
    cv2.rectangle(clone, (x, y), (x + window_size[0], y + window_size[1]), color, thickness = 2)
    imgutil.show_img((clone,), ("Sliding Window in Progress",), 1)

  def get_init_factor(self, img, scaled):
    w_factor = float(img.shape[1]) / float(scaled.shape[1])
    h_factor = float(img.shape[0]) / float(scaled.shape[0])
    return max(w_factor, h_factor)


  def detect_multi_scale(self, img, p):
    window_size = self.hog.get_parameter('window_size')
    step_size = self.hog.get_parameter('step_size')
    scale = p['scale']
    detected_window = []
    #looks like we have too many matches. we only need one
    #from smallest scale to original scale
    reversed=False; max_margin = 0; factor = 0
    for (i, scaled)  in enumerate(imgutil.pyramid(img, scale=scale, minSize=window_size, reversed=reversed)):
      if i == 0:
        factor = self.get_init_factor(img, scaled)
      elif reversed:
        factor = factor/scale
      else:
        factor = factor * scale
      t2 = time.time()
      for (x, y, window) in  imgutil.sliding_window(scaled, window_size, step_size):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
          continue
        if p['visual_scan']:
          self.show_window(scaled, (x,y), window_size)
        t0 = time.time()
        vec = self.hog.extract(window)
        t1 = time.time()
        if self.clf.predict(vec):
          margin = self.clf.decision_function(vec)
          logging.debug("found object with %(margin)f" % locals())
          #TODO: why?
          if margin < 0.30:
            logging.debug("skipping detection with low margin %(margin)f" % locals())
            continue
          if margin > max_margin:
            max_margin = margin
          #to scale back to original size
          detected_window.append((int(x * factor), int(y * factor), margin, int(window_size[0] * factor),
                                  int(window_size[1] * factor)))
          if p['visual_scan'] or p['visual_detection']:
            self.show_window(scaled, (x,y), window_size, color=(0,255,0))
      logging.debug("scanning level %d (%d,%d) takes %.2f" % (i, window.shape[1], window.shape[0], time.time()-t2))
      #TODO: why?
      if max_margin > 1.5:
        break
    return (detected_window, max_margin)

  def detect_img(self, image_name, output_path, p):
    assert os.path.isfile(image_name)
    logging.debug("scanning %s" % image_name)
    window_size = self.hog.get_parameter('window_size')
    shrink_w_size = int(p['shrink_w_size'])

    origin = cv2.imread(image_name)
    resized = imgutil.resize(origin, width=shrink_w_size, height=shrink_w_size*2)
    assert resized is not None and resized.shape[1]>=window_size[0] and resized.shape[0] >= window_size[1]

    detected_window = [] ; best_img = None; best_margin = 0; best_window = []
    for degree in (0,):
      logging.debug("scanning with %(degree)d degree" % locals())
      #rotate and gray scale before processing
      rotated = imgutil.rotate(resized.copy(), degree)
      gray= cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
      (detected_window, margin) = self.detect_multi_scale(gray, p)
      if margin > best_margin:
        best_margin = margin
        best_img = rotated
        best_window = detected_window
      #TODO: why
      if len(best_window) > 0 and best_margin > 5.0:
        break
    #Non-Maximum Suppression
    if ( len(best_window) > 1 ):
      detected_window = nms.nms(best_window, p['threshold'])



    if output_path is not None and len(detected_window) > 0:
      self.write_detected_img(best_img, detected_window, image_name, output_path, p)
    return len(detected_window) > 0

  def detect(self, args):
    img_path = args['img_path']
    output_path = args['output_path']
    parameters = {}
    for k in ('once', 'visual_scan', 'visual_detection'):
      parameters[k] = args[k] if args[k] is not None else False

    for k in ('shrink_w_size', 'scale', 'threshold'):
      parameters[k] = float(args[k]) if args[k] is not None else self.hog.get_parameter(k)
      logging.info("setting %s to %f", k, parameters[k])
    assert os.path.isdir(img_path)
    match = 0; cnt = 0
    t0 = time.time()
    neg_images = []
    for subdir, dirs, files in os.walk(img_path):
      for basename in files:
        m = re.match(r'.*[.]jpg', basename)
        if m is None: continue
        filename = os.path.join(subdir, basename)
        if self.detect_img(filename, output_path, parameters):
          match += 1
        else:
          neg_images.append(filename)
        cnt += 1
        n = cnt - match
        logging.info("P=%(match)d, N=%(n)d, cnt=%(cnt)d" %locals())
    t = time.time() - t0
    n = cnt - match
    logging.info("P=%(match)d, N=%(n)d, total images=%(cnt)d images, in %(t).2f seconds" %locals())
    logging.info("N: %s", " ".join([ f for f in  neg_images ]) )


def process(args):
  detector = HogDetector(args)
  if args['action'] == 'train':
    pass
  elif args['action'] == 'detect':
    detector.detect(args)
  else:
    raise ValueError('unknown action')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, version=__version__)
  parser.add_argument('-l', dest='logLevel', default='info', help="logging level: {debug, info, error}")
  parser.add_argument('--train_file', dest='train_file', help="supply feature vectors for training")
  parser.add_argument('--img_path', dest='img_path', help="supply image path for leaf detection")
  parser.add_argument('--output_path', dest='output_path', help="supply output path for leaf detection")
  parser.add_argument('--only_once', dest='once', default=False, action='store_true',
                      help="write only one detection per image")
  parser.add_argument('--visual_scan', dest='visual_scan', default=False, action='store_true',
                      help="visualize scanning process ")
  parser.add_argument('--visual_detection', dest='visual_detection', default=False, action='store_true',
                      help="visualize scanning detection ")
  parser.add_argument('action', help="perform action {train, detect}")
  parser.add_argument('model_file', help="supply filename for model persistence")
  for k in ('shrink_w_size', 'scale', 'threshold'):
    parser.add_argument('--'+ k , dest=k, help='customize detection parameters '+k)
  args = parser.parse_args()
  logging.basicConfig(level=getattr(logging, args.logLevel.upper()), format='%(asctime)s %(levelname)s %(message)s')

  process(vars(args))



