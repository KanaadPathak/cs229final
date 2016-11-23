#!/usr/bin/env python
"""

"""
__version__ = "0.2"
import argparse, sys, os
import blosc
import cv2
import logging
import numpy as np
import pickle
import random
import re
import xml.etree.ElementTree as ET
from abc import ABCMeta, abstractmethod


class ImageRecordSerializer(object):
  """
  Image Records is a list of ( record#1, record#2, ..., record#n)
  Each element is a tuple of
    (species, original image name, np array of an image)
  Optionally, meta info can be stored as well
  """
  @classmethod
  def serialize(cls, filename, records, meta_info=None):
    """ write image records to file
    filename: full filename to file to which the records will be written
    """
    (labels, image_names, images) = zip(*records)
    d = {
        'labels' : labels,
        'image_names' : image_names,
        'images' : images,
        'meta_info' : meta_info,
        }
    with open(filename, 'w') as fd:
      pickle.dump(d, fd)

  @classmethod
  def deserialize(cls, filename):
    """read image records from file
    filename: full filename to the file from which records will be read
    output: records
    """
    (records, meta_info) = cls.deserialize_with_meta(filename)
    return records

  @classmethod
  def deserialize_with_meta(cls, filename):
    """read image records from file
    filename: full filename to the file from which records will be read
    output: (records, meta_info)
    """
    with open(filename, 'r') as fd:
      d = pickle.load(fd)
    meta_info = d['meta_info'] if 'meta_info' in d else None
    return (zip(d['labels'], d['image_names'], d['images']), meta_info)

class LeafPreprocessor(object):
  __metaclass__ = ABCMeta
  @abstractmethod
  def read_record(self, filename, basename):
    """ return (valid, image, label) from the file, if any
    path: path to the file
    basename: basename
    valid: true if the image should be processed
    image: cropped image array
    label: integer label
    """
    pass

  @abstractmethod
  def verify_record(self, label, name, image):
    """ verify image record"""
    return False

  def process(self, path):
    """ iterate all images in path and return nparray
    output: list of (species, original image name, np array of an image)
    """
    assert os.path.exists(path), "%s doesn't exist"%(path)
    labels=[]; image_names=[]; images=[]; 
    for subdir, dirs, files in os.walk(path):
      for f in files:
        filename = os.path.join(subdir,f)
        logging.debug( "working on %s" % filename)
        (valid, img, label) = self.read_record(subdir, f)
        if not valid:
          logging.debug("ignoring %s" % filename)
          continue
        assert img is not None, "invalid image:%s"%(filename)
        labels.append(label); image_names.append(filename); images.append(img)
    self.records =  zip(labels, image_names, images)
    return (self.records, self.meta_info)

  def split(self, split_ratio):
    """randomly shuffle and split train/test data
    split_ratio: ratio of the split and must be in  (0, 1)
    """
    assert split_ratio<1.0 and split_ratio>0, "Wrong split:%f"%split_ratio
    records1 = []; records2 = []; dict = {}
    for record in self.records:
      label = record[0]
      if label not in dict:
        dict[label] = []
      dict[label].append(record)
    for (label, records) in dict.items():
      random.shuffle(records)
      total_size = len(records)
      split_size = int(total_size*split_ratio)
      logging.debug("splitting label:%(label)d %(split_size)d out of %(total_size)d"%locals())
      records1 += records[:split_size]
      records2 += records[split_size:]
    return (records1, records2)

  def resize(self, img, w_reduced):
    """ helper method to downsize image"""
    (h, w) = img.shape
    w = min(w_reduced, w)
    dim = (int(w), int( img.shape[0] * w/img.shape[1]))
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


  def verify(self, filename):
    (records, self.meta_info) = ImageRecordSerializer.deserialize_with_meta(filename)
    logging.info("%s: total records %d" % (filename, len(records)))
    for (i, record) in enumerate(records):
      assert len(record)==3, "record %d: wrong size: %d"%(i, len(record))
      (label, name, image) = record
      logging.debug("record%d: label=%d image_name=%s" % (i, label, name))
      assert os.path.exists(name) and os.path.isfile(name), "record %d: %s not a file"%(i, name)
      try:
        self.verify_record(label, name, image)
      except:
        raise "record %d: doesn't match"%(i)

  def __init__(self):
    self.meta_info = None
    self.records = None

class SwedishLP(LeafPreprocessor):
  """ Swedish data set"""
  def downsize(self, img):
    #SIFT can still give ~200 KP per image, verified manually on a few images
    return self.resize(img, 128.0)

  def read_record(self, path, basename):
    m = re.match(r"l([0-9]+)nr[0-9]*[.]tif", basename)
    if m is None:
      return (False, None, None)
    label = int(m.group(1))
    #read image
    filename = os.path.join(path, basename)
    img = cv2.imread(filename,0);
    img = self.downsize(img)
    assert img is not None, "cannot read image:%s"%(basename)
    return (True, img, label)

  def verify_record(self, label_expected, name, image_expected):
    base = os.path.basename(name)
    path = os.path.dirname(name)
    (valid, img, label) = self.read_record(path, base)
    assert label_expected==label, "%(lable)s doesn't match %(label_expected)s" % locals()
    assert img is not None, "cannot read image:%s"%(name)
    img = self.downsize(img)
    assert np.sum(np.where(image_expected!=img))==0, "image doesn't match"

class ImageCLEF2013(LeafPreprocessor):
  """ ImageCLEF2013 format
  All images are in jpeg format and were resized so that the maximum width or height does not exceed 800 pixels.
  Each image is uniquely identified by an integer "uid" between 1 and 30000.
  """
  def read_property_xml(self, path, basename):
    m = re.match(r"([0-9]+)[.]jpg", basename)
    if m is None:
      return None
    #read property xml
    xml_f = os.path.join( path, m.group(1) + '.xml')
    if not (os.path.exists(xml_f) and os.path.isfile(xml_f)):
      raise ValueError("jpg without property xml")
    with open(xml_f, 'r') as fd:
      tree = ET.fromstring(fd.read())
    return tree

  def downsize(self, img):
    #~300 KP with original size and SIFT detector. requires dense detector
    return self.resize(img, 128.0)

  def read_record(self, path, basename):
    tree = self.read_property_xml(path, basename)
    if tree is None:
      return (False, None, None)
    #filter out files we don't like
    for (k, r) in self.required_properties.items():
      v = tree.find(k).text
      logging.debug("k:%(k)s r:%(r)s v:%(v)s" % locals())
      if v.lower() != r.lower():
        return (False, None, None)
    #read label
    label_name = tree.find('ClassId').text
    dict = self.meta_info['label_names']
    if label_name not in dict:
      dict[label_name] = len(dict) + 1
    label = dict[label_name]
    #read image
    img = self.downsize(cv2.imread(os.path.join(path, basename), 0))
    assert img is not None, "cannot read image:%s"%(basename)
    return (True, img, label)

  def verify_record(self, label_expected, name, image_expected):
    basename = os.path.basename(name)
    path = os.path.dirname(name)
    (valid, img, label) = self.read_record(path, basename)
    assert valid and img is not None, 'Image is not valid'
    tree = self.read_property_xml(path, basename)
    species = tree.find('ClassId').text
    assert species in self.meta_info['label_names'], 'species name is not in meta info'
    assert label_expected == self.meta_info['label_names'][species], \
      "%(lable)s doesn't match %(label_expected)s" % locals()
    img = self.downsize(img)
    assert np.sum(np.where(image_expected!=img))==0, "image doesn't match"

  def __init__(self, required_properties):
    """"
    required_properties
    -Acquisition Type: {SheetAsBackground, NaturalBackground}
    #SheetAsBackground: uniform background (42%): exclusively pictures of leaves in front of a white or colored uniform background
    produced with a scanner or a camera with a sheet
    #NaturalBackground: for most of the time cluttered natural background (58%): free natural photographs of different views
    on different subparts of a plant into the wild.
    -View Content: {Leaf, Flower, Fruit, Stem, Entire }
    """
    LeafPreprocessor.__init__(self)
    #store label labels dictionary { "label_name" : label }
    self.meta_info = {'label_names':{}}
    self.required_properties = required_properties

def make_preprocessor(type):
  if (type == 'swedish'):
    return SwedishLP()
  elif (type == 'clef2013_uniform_leaf'):
    return ImageCLEF2013({'Type':'SheetAsBackground', 'Content':'Leaf'})
  raise ValueError("Unknown data set type:%s"% (type))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, version=__version__)
  parser.add_argument('-l', dest='logLevel', default='info', help="logging level: {debug, info, error}")
  parser.add_argument('--verify,', dest='verify', action='store_true', help = "verify")
  parser.add_argument('--split_output,', dest='split_output', help = "split records into two file per split_ratio")
  parser.add_argument('--split_ratio,', dest='split_ratio', default=0.7, type=float,
      help = "percentage of total data used for training")
  parser.add_argument('type', help = "supply dataset type: {swedish}")
  parser.add_argument('img_path', help = "supply path to database images")
  parser.add_argument('record_file', help = "supply filename for preprocessed records")
  args = parser.parse_args()

  logging.basicConfig(level=getattr(logging, args.logLevel.upper()),
                      format='%(asctime)s %(levelname)s %(message)s')

  p = make_preprocessor(args.type)
  if args.verify:
    logging.info("verifying %s" % args.record_file)
    p.verify(args.record_file)
    logging.info("verified successfully!")
    sys.exit()

  for f in (args.record_file, args.split_output):
    if f is not None and os.path.exists(f):
      logging.error("Refuse to overwrite existing file: %s" % f)
      sys.exit()

  (records, meta_info) = p.process(args.img_path)
  logging.info("total record size:%d", len(records))

  if args.split_output is not None:
    logging.info("splitting records to %s and %s with ratio:%f", args.record_file, args.split_output, args.split_ratio)
    (records, split_records) = p.split(args.split_ratio)
    logging.info("writing %d records to  %s" % (len(split_records),  args.split_output))
    ImageRecordSerializer.serialize(args.split_output, split_records, meta_info)
  logging.info("writing %d records to  %s" % (len(records),  args.record_file))
  ImageRecordSerializer.serialize(args.record_file, records, meta_info)
