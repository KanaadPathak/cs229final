#!/usr/bin/env python

import os
from xml.dom import minidom
import shutil
import sys


def extract_leaves(root_dir):
    # root_dir = sys.argv[1]
    image_dir = '%s/train' % root_dir
    for f in os.listdir(image_dir):
        if f.endswith('.xml'):
            filename = '%s/%s' % (image_dir, f)
            xmldoc = minidom.parse(filename)
            the_content = xmldoc.getElementsByTagName('Content')[0].firstChild.data
            the_type = xmldoc.getElementsByTagName('Type')[0].firstChild.data
            the_classid = xmldoc.getElementsByTagName('ClassId')[0].firstChild.data
            the_filename = xmldoc.getElementsByTagName('FileName')[0].firstChild.data
            if the_content == 'Leaf':
                print(','.join([the_type, the_classid, the_content, the_filename]))
                dest_dir = '%s/ready' % root_dir
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy('%s/%s' % (image_dir, the_filename), '%s/%s' % (dest_dir, the_filename))

if __name__ == '__main__':
    extract_leaves(sys.argv[1])
