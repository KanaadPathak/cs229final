#!/usr/bin/env python
"""

"""
__version__ = "0.2"
import argparse, sys, os
import cv2
import image_feature
import imgutil
import logging
import numpy as np
import json
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from sklearn.externals import joblib

def process(mask_path, input_path, output_path):
  for subdir, dirs, files in os.walk(mask_path):
    for f in files:
      m = re.match(r"(.*)[.]jpg", f)
      if m is None:
        continue
      filename = os.path.join(input_path,f)
      logging.info("copying %(filename)s" % locals())
      shutil.copy(filename, output_path)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, version=__version__)
  parser.add_argument('-l', dest='logLevel', default='info', help="logging level: {debug, info, error}")
  parser.add_argument('mask_path', help = "supply path to images names to be copied")
  parser.add_argument('input_path', help = "supply path to database images")
  parser.add_argument('output_path', help = "supply path to output images")
  args = parser.parse_args()

  logging.basicConfig(level=getattr(logging, args.logLevel.upper()), format='%(asctime)s %(levelname)s %(message)s')
  process(args.mask_path, args.input_path, args.output_path)
