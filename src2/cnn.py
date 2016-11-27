import argparse
from datetime import datetime

import yaml
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Dropout, Flatten, Dense
from keras.models import Sequential

from cnn_preprocess import count_image, get_classes, load_generator


class CNNClassifier(object):
    def __init__(self, num_classes=2, target_size=(255, 255)):
        # dimensions of our images.
        self.img_width, self.img_height = target_size

        self.num_classes = num_classes
        self.model = self.create_model()

    def create_model(self):
        loss = 'binary_crossentropy'
        output_dim = 1
        final_activation = 'sigmoid'
        if self.num_classes > 2:
            loss = 'categorical_crossentropy'
            output_dim = self.num_classes
            final_activation = 'softmax'

        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(3, self.img_height, self.img_width)))
        model.add(Convolution2D(32, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim, activation=final_activation))

        model.compile(loss=loss,
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        return model

    def fit_generator(self, train_generator, val_generator, num_train, num_val, num_epoch=50):
        self.model.fit_generator(
            train_generator,
            samples_per_epoch=num_train,
            nb_epoch=num_epoch,
            validation_data=val_generator,
            nb_val_samples=num_val)

    def save(self, h5_file):
        self.model.save_weights(h5_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--epoch', type=int, default=50, help="the number of epochs to run")
    parser.add_argument('-s', '--save_file', help="the file that the weight are saved to")
    parser.add_argument('config_file', help="the path to the config")
    args = parser.parse_args()

    with open(args.config_file, 'r') as stream:
        conf = yaml.load(stream)

    batch_size = conf.get('batch_size', 32)
    target_size = (conf.get('img_height', 256), conf.get('img_width', 256))
    train_data_dir = '%s/train' % conf.get('data_dir')
    val_data_dir = '%s/validation' % conf.get('data_dir')
    train_gen_conf = conf.get('train_gen', {})
    val_gen_conf = conf.get('val_gen', {})

    train_gen = load_generator(train_data_dir, target_size, batch_size, **train_gen_conf)
    val_gen = load_generator(val_data_dir, target_size, batch_size, **val_gen_conf)

    clf = CNNClassifier(num_classes=train_gen.nb_class, target_size=target_size)
    clf.fit_generator(train_gen, val_gen, num_train=train_gen.nb_sample, num_val=val_gen.nb_sample,
                      num_epoch=args.epoch)

    if args.save_file is not None:
        clf.save(args.save_file)
    else:
        filename = args.config_file.rsplit('/', 1)[-1]
        name = filename.rsplit('.', 1)[0]
        dt = datetime.now().strftime("%Y%m%d%H%M%S")
        save_file = 'history/%s-%s.h5' % (name, dt)
        clf.save(save_file)
