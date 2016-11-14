#!/usr/bin/env python
"""

"""
__version__ = "0.1"
import argparse, sys, os
import cv2
import logging
import numpy as np
import pickle
import random
import re
from abc import ABCMeta, abstractmethod

class ImageRecordSerializer(object):
  """
  Image Records is a list of ( record#1, record#2, ..., record#n)
  Each element is a tuple of
    (species, original image name, np array of an image)
  """
  @classmethod
  def serialize(cls, filename, records):
    """ write image records to file
    filename: file the records will be written to
    """
    (labels, image_names, images) = zip(*records)
    d = {
        'labels' : labels,
        'image_names' : image_names,
        'images' : images,
        }
    with open(filename, 'w') as fd:
      pickle.dump(d, fd)

  @classmethod
  def deserialize(cls, filename):
    """read image records from file
    filename: file the records will be written to
    output: records
    """
    with open(filename, 'r') as fd:
      d = pickle.load(fd)
    return zip(d['labels'], d['image_names'], d['images'])

class LeafPreprocessor(object):
  __metaclass__ = ABCMeta
  @abstractmethod
  def parse_labels(self, filename):
    """ return the lable of the image """
    pass

  @abstractmethod
  def scale(self, img):
    """ return the scaled image """
    pass

  @abstractmethod
  def get_filemask(self):
    """ return the mask for the file, if any """
    pass

  def process(self, path):
    """ iterate all images in path and return nparray
    output: list of (species, original image name, np array of an image)
    """
    assert os.path.exists(path), "%s doesn't exist"%(path)
    labels=[]; image_names=[]; images=[]; 
    for subdir, dirs, files in os.walk(path):
      for f in files:
        filename = os.path.join(subdir,f)
        logging.debug(filename)
        filemask = self.get_filemask()
        if filemask and not filemask.match(f):
          logging.warning("ignoring %s" % f)
          continue
        img = cv2.imread(filename,0); 
        assert img is not None, "cannot read image:%s"%(filename)
        img = self.scale(img)
        label = self.parse_labels(f)
        labels.append(label); image_names.append(filename); images.append(img)
    return zip(labels, image_names, images)

  def verify(self, filename):
    assert input, "empty input"
    for (i,r) in enumerate(ImageRecordSerializer.deserialize(filename)):
      assert len(r)==3, "record %d: wrong size: %d"%(i, len(r))
      (label, name, image) = r
      logging.debug("record%d: label=%d image_name=%s" % (i, label, name))
      assert os.path.exists(name) and os.path.isfile(name), "record %d: %s not a file"%(i, name)
      assert self.parse_labels(os.path.basename(name))==label, "record %d: lable is wrong" %(i)
      img = cv2.imread(name,0)
      assert img is not None, "record %d: cannot read image:%s"%(i, name)
      img = self.scale(img)
      assert np.sum(np.where(image!=img))==0, "record %d: image doesn't match"%(i)



class SwedishLP(LeafPreprocessor):
  def parse_labels(self, filename):
    m = re.match(r"l([0-9]+)nr.*", filename)
    assert m.group(0)
    return int(m.group(1))

  def scale(self, img):
    """ return scaled """
    #imread return array as (height,  width, ...) 
    w = 128.0
    dim = (int(w), int( img.shape[0] * w/img.shape[1]))
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

  def get_filemask(self):
    """ return the mask for the file, if any """
    return re.compile(r"l[0-9]+nr[0-9]+[.]tif")

def make_preprocessor(type):
  if (type == 'swedish'):
    return SwedishLP()
  raise ValueError("Unknown dataset type:%s"% (type))



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, version=__version__)
  parser.add_argument('-l', dest='logLevel', default='info',
      help="logging level: {debug, info, error}")
  parser.add_argument('--verify,', dest='verify', action='store_true', 
      help = "verify preprocessing")
  parser.add_argument('--train_size,', dest='train_size', default=0.7, type=float, 
      help = "percentage of total data used for training")
  parser.add_argument('type', help = "supply dataset type: {swedish}")
  parser.add_argument('path', help = "supply path to database images")
  parser.add_argument('train_file', help = "supply filename for preprocessed training data")
  parser.add_argument('test_file', help = "supply filename for preprocessed test data")
  args = parser.parse_args()

  logging.basicConfig(level=getattr(logging, args.logLevel.upper()),
                      format='%(asctime)s %(levelname)s %(message)s')

  p = make_preprocessor(args.type)
  if args.verify:
    for f in (args.train_file, args.test_file):
      logging.info("verifying %s"%f)
      p.verify(f)
    sys.exit()

  for f in (args.train_file, args.test_file):
    if os.path.exists(f):
      logging.error("Refuse to overwrite existing file: %s" % f)
      sys.exit()

  records = p.process(args.path)
  random.shuffle(records)

  assert args.train_size<=1.0 and args.train_size>0, "Wrong split:%f"%train_size
  train_size = int(len(records)*args.train_size)
  logging.debug("train size:%d total:%d", train_size, len(records));

  logging.info("writting train data to %s"%args.train_file)
  ImageRecordSerializer.serialize(args.train_file, records[:train_size])
  logging.info("writting test data to %s"%args.test_file)
  ImageRecordSerializer.serialize(args.test_file, records[train_size:])
