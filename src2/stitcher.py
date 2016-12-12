import argparse

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import cv2
import os
import sys
import numpy as np


def _convert(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def stitch(img_path, target_size=(128, 192), rows=0, cols=0):
    jpgs = [f for f in os.listdir(img_path) if f.endswith('jpeg') or f.endswith('jpg')]
    first_path = os.path.join(img_path, jpgs[0])
    img = cv2.imread(first_path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        img_width, img_height = img.shape
        channel = 1
    else:
        img_width, img_height, channel = img.shape
    target_channel = 3

    name = os.path.dirname(img_path).split('/')[-1]
    rows = rows if rows != 0 else int(round(target_size[0] / img_height))
    cols = cols if cols != 0 else int(round(target_size[1] / img_width))
    print(rows, cols, name)

    margin = 1
    height = rows * img_height + (rows - 1) * margin
    width = cols * img_width + (cols - 1) * margin
    stitched_filters = np.ones((height, width, target_channel)) * 128

    # fill the picture with our saved filters
    jpgs = [f for f in os.listdir(img_path) if f.endswith('jpeg') or f.endswith('jpg')]
    for i in range(rows):
        for j in range(cols):
            f = jpgs[i * cols + j]
            print('stitching %s' % f)
            img = cv2.imread(os.path.join(img_path, f), cv2.IMREAD_UNCHANGED)
            if target_channel == 3 and channel == 1:
                white = np.max(img)
                black = np.min(img)
                img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
                img * (white-black) / 255 + black
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    cv2.imwrite('%s.jpg' % name, stitched_filters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help="the path to the config file")
    parser.add_argument('-r', '--rows', type=int, default=0, help="number of rows")
    parser.add_argument('-c', '--cols', type=int, default=0, help="number of column")
    args = parser.parse_args()
    stitch(args.dir, rows=args.rows, cols=args.cols)
