import argparse

from functools import reduce

from cnn_preprocess import load_generator, count_image
from vgg16 import VGG16
from tqdm import tqdm
from operator import mul
import tables
# import numpy as np

__version__ = "0.1"

class CNNFeatureExtractor(object):
    def __init__(self):
        pass

    def extract_feature(self, data_dir, feature_file, target_size=(256, 256), batch_size=32):
        model = VGG16(weights='imagenet', include_top=False)
        data_gen = load_generator(data_dir, target_size=target_size, batch_size=batch_size)

        with tqdm(total=data_gen.nb_sample) as pbar, tables.open_file(feature_file, mode='w') as f:
            atom = tables.Float64Atom()
            cnn_input_shape = (batch_size, *data_gen.image_shape)
            cnn_output_shape = model.get_output_shape_for(cnn_input_shape)
            feature_shape = (batch_size, reduce(mul, cnn_output_shape[1:]))
            features = f.create_earray(f.root, 'features', atom, (0, feature_shape[1]))
            labels = f.create_earray(f.root, 'labels', atom, (0, data_gen.nb_class))

            for x, y in data_gen:
                new_x = model.predict(x)
                new_x = new_x.reshape(feature_shape)
                features.append(new_x)
                labels.append(y)
                pbar.update(batch_size)

    def load_features(self, feature_file):
        with tables.open_file(feature_file, mode='r') as f:
            print(f.root.data[:2, :])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-V', '--version', action='version', version=__version__)
    parser.add_argument('-s', '--feature_file', help="the file that the weight are saved to")
    parser.add_argument('data_dir', help="the path to the config")
    args = parser.parse_args()

    if args.feature_file is not None:
        feature_file = args.save_file
    else:
        feature_file = 'default.h5'

    CNNFeatureExtractor().extract_feature(args.data_dir, feature_file)


