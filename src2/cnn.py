from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from argparse import ArgumentParser
import yaml
import os
from datetime import datetime


class CNNClassifier(object):
    def __init__(self, num_classes=2, img_height=150, img_width=150):
        # dimensions of our images.
        self.img_width = img_width
        self.img_height = img_height

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

    def vgg16(self):
        loss = 'binary_crossentropy'
        output_dim = 1
        final_activation = 'sigmoid'
        if self.num_classes > 2:
            loss = 'categorical_crossentropy'
            output_dim = self.num_classes
            final_activation = 'softmax'

        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(3, self.img_height, self.img_width)))

        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim, activation=final_activation))

        model.compile(optimizer='rmsprop', loss=loss, metrics=['accuracy'])
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


def get_classes(data_dir):
    classes_ = []
    for a_class in os.listdir(data_dir):
        class_dir = '%s/%s' % (data_dir, a_class)
        if os.path.isdir(class_dir):
            classes_.append(a_class)
    return classes_


def count_image(data_dir):
    supported_ext = ('.jpeg', '.jpg', '.png')
    image_count = 0
    for root, dirs, files in os.walk(data_dir):
        image_count += sum(f.endswith(supported_ext) for f in files)
    return image_count


def load_generator(data_dir, target_size, batch_size, generator_params):
    classes_ = get_classes(data_dir)
    class_mode = 'binary' if len(classes) <= 2 else 'categorical'

    data_gen = ImageDataGenerator(**generator_params)
    return data_gen.flow_from_directory(
        data_dir,
        target_size=target_size,
        classes=classes_,
        class_mode=class_mode,
        batch_size=batch_size)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--epoch', type=int, default=50, help="the number of epochs to run")
    parser.add_argument('-s', '--save_file', help="the file that the weight are saved to")
    parser.add_argument('config_file', help="the path to the config")
    args = parser.parse_args()

    with open(args.config_file, 'r') as stream:
        conf = yaml.load(stream)

    data_dir = conf.get('data_dir')
    img_height = conf.get('img_height', 256)
    img_width = conf.get('img_width', 256)
    batch_size = conf.get('batch_size', 32)
    target_size = (img_height, img_width)
    train_data_dir = '%s/train' % data_dir
    val_data_dir = '%s/validation' % data_dir
    classes = get_classes(train_data_dir)
    num_train = conf.get('num_train', count_image(train_data_dir))
    num_test = conf.get('num_test', count_image(val_data_dir))
    train_gen_conf = conf.get('train_gen', {})
    val_gen_conf = conf.get('val_gen', {})

    clf = CNNClassifier(num_classes=len(classes), img_height=img_height, img_width=img_width)
    clf.fit_generator(load_generator(train_data_dir, target_size, batch_size, train_gen_conf),
                      load_generator(val_data_dir, target_size, batch_size, val_gen_conf),
                      num_train=num_train,
                      num_val=num_test,
                      num_epoch=args.epoch)

    if args.save_file is not None:
        clf.save(args.save_file)
    else:
        filename = args.config_file.rsplit('/', 1)[-1]
        name = filename.rsplit('.', 1)[0]
        dt = datetime.now().strftime("%Y%m%d%H%M%S")
        save_file = 'history/%s-%s.h5' % (name, dt)
        clf.save(save_file)
