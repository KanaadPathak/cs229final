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


def stitch(img_path, target_size=(256, 256), rows=0, cols=0):
    jpgs = [f for f in os.listdir(img_path) if f.endswith('jpeg') or f.endswith('jpg')]
    first_path = os.path.join(img_path, jpgs[0])
    img = cv2.imread(first_path, cv2.IMREAD_UNCHANGED)
    img_width, img_height, channel = img.shape

    name = os.path.dirname(img_path).split('/')[-1]
    cols = cols if cols != 0 else int(round(target_size[0] / img_width))
    rows = rows if rows != 0 else int(round(target_size[1] / img_height))
    print(rows, cols, name)

    margin = 1
    width = cols * img_width + (cols - 1) * margin
    height = rows * img_height + (rows - 1) * margin
    stitched_filters = np.zeros((width, height, channel))

    # fill the picture with our saved filters
    jpgs = [f for f in os.listdir(img_path) if f.endswith('jpeg') or f.endswith('jpg')]
    for i in range(cols):
        for j in range(rows):
            f = jpgs[i * cols + j]
            print('stitching %s' % f)
            img = cv2.imread(os.path.join(img_path, f), cv2.IMREAD_UNCHANGED)
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    cv2.imwrite('%s.jpg' % name, stitched_filters)


stitch(sys.argv[1], rows=3, cols=2)
