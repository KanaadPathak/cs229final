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
import sys
import xml.etree.ElementTree as ET
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from sklearn.externals import joblib

filenames = []; content = []
with open(sys.argv[1], 'r') as fd:
  for line in fd:
    items = line.split(' ')
    filenames.append(items[0])
    content.append(items[1:])
del filenames[0]
with open(sys.argv[2], 'w') as fd:
  for (f, c) in zip(filenames, content):
    fd.write( ' '.join([f] + c ) )



