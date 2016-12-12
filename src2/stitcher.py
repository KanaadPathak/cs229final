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


def stitch(img_path, target_size=(128, 128)):
    jpgs = [f for f in os.listdir(img_path) if f.endswith('jpeg') or f.endswith('jpg')]
    first_path = os.path.join(img_path, jpgs[0])
    img = cv2.imread(first_path, cv2.IMREAD_UNCHANGED)
    img_width, img_height, channel = img.shape

    name = os.path.dirname(img_path).split('/')[-1]
    n = int(round(target_size[0]/img_width))
    print(n, name)

    margin = 1
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, channel))

    # fill the picture with our saved filters
    jpgs = [f for f in os.listdir(img_path) if f.endswith('jpeg') or f.endswith('jpg')]
    for i in range(n):
        for j in range(n):
            f = jpgs[i * n + j]
            print('stitching %s' % f)
            img = cv2.imread(os.path.join(img_path, f), cv2.IMREAD_UNCHANGED)
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    cv2.imwrite('%s.jpg' % name, stitched_filters)


stitch(sys.argv[1])
