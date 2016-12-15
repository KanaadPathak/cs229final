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

def shift_color(x, factor):
  assert factor <= 1.0 and factor > 0.0
  channel_index = 2
  mask = x > 10
  x = np.rollaxis(x, channel_index, 0)
  #fg = np.partition(x.flatten(), len(x.flatten())-1)
  #min_x = fg[np.where(fg)[0][0]]
  intensity = (x.max() - x.mean()) * factor
  channel_images = []
  for x_channel in x:
    channel_images.append(np.clip(x_channel + np.random.uniform(0, intensity), 0, 255))
  x = np.stack(channel_images, axis=0)
  x = np.rollaxis(x, 0, channel_index + 1)
  x = x * (mask.astype(np.uint8))
  return x.astype(np.uint8)

def random_scale(x, scale):
  expected_size = 256
  assert scale<=1.0
  scale_factor = np.random.uniform(scale, 1.0)
  orig_h = x.shape[0]
  orig_w = x.shape[1]
  if orig_h != expected_size or orig_w !=expected_size:
    x = imgutil.resize(x, expected_size, expected_size)
  new_size = int(expected_size * scale_factor)
  x = imgutil.shrink(x, new_size, new_size)
  x = imgutil.find_surrounding_box(x, expected_size, expected_size)
  return x

def process(args):
  img_path = args['img_path']
  output_root = args['output_path']
  factor = args['factor']
  intensity = args['intensity']
  scale = args['scale']
  debug_images = None
  for subdir, dirs, files in os.walk(img_path):
    for f in files:
      m = re.match(r"(.*)[.]jpg", f)
      if m is None:
        continue
      filename = os.path.join(subdir,f)
      logging.debug("working on %(filename)s" % locals())
      img = cv2.imread(filename)
      output_path = os.path.join(output_root, os.path.basename(subdir))
      if not os.path.exists(output_path):
        os.mkdir(output_path)
      for i in range(factor):
        if i != 0:
          img = random_scale(img, scale)
          img = shift_color(img, intensity)
        output_filename = os.path.join(output_path, m.group(1) + '_%d.jpg'%(i))
        assert cv2.imwrite(output_filename, img)
        logging.debug("writing to %(output_filename)s" % locals())

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, version=__version__)
  parser.add_argument('-l', dest='logLevel', default='info', help="logging level: {debug, info, error}")
  parser.add_argument('--factor', dest="factor", type=int, default=10, help = "factor to augment")
  parser.add_argument('--intensity', dest="intensity", type=float, default=0.2, help = "channel shift intensity factor")
  parser.add_argument('--scale', dest="scale", type=float, default=0.8, help = "factor to scale down")
  parser.add_argument('img_path', help = "supply path to database images")
  parser.add_argument('output_path', help = "supply path to output images")
  args = parser.parse_args()

  logging.basicConfig(level=getattr(logging, args.logLevel.upper()), format='%(asctime)s %(levelname)s %(message)s')
  process(vars(args))




