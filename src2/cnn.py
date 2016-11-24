from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from argparse import ArgumentParser
import yaml
import os

class CNNClassifier(object):
    def __init__(self, num_classes=2, img_height=150, img_width=150):
        # dimensions of our images.
        self.img_width = img_width
        self.img_height = img_height

        self.num_classes = num_classes
        self.model = self.create_model()

    def create_model(self):
        loss = 'binary_crossentropy'
        final_activation = 'sigmoid'
        if self.num_classes > 2:
            loss = 'categorical_crossentropy'
            final_activation = 'softmax'

        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape=(3, self.img_height, self.img_width)))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1 if self.num_classes == 2 else self.num_classes))
        model.add(Activation(final_activation))

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


def make_divisible(total, divisor):
    remainder = total % divisor
    if remainder != 0:
        return total + divisor - remainder
    else:
        return total


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--epoch', type=int, default=50, help="the number of epochs to run")
    parser.add_argument('-s', '--save_file', help="the file that the weight are saved to")
    parser.add_argument('config_file', help="the path to the config")
    args = parser.parse_args()

    with open(args.config_file, 'r') as stream:
        conf = yaml.load(stream)

    img_height = conf['img_height']
    img_width = conf['img_width']
    target_size = (img_height, img_width)
    batch_size = conf['batch_size']
    train_data_dir = '%s/train' % conf['data_dir']
    val_data_dir = '%s/validation' % conf['data_dir']

    if 'num_train' in conf:
        num_train = conf['num_train']
    else:
        num_train = make_divisible(count_image(train_data_dir), batch_size)

    if 'num_test' in conf:
        num_test = conf['num_test']
    else:
        num_test = make_divisible(count_image(val_data_dir), batch_size)

    train_gen_conf = conf['train_gen']
    val_gen_conf = conf['val_gen']

    classes = get_classes(train_data_dir)

    clf = CNNClassifier(num_classes=len(classes), img_height=img_height, img_width=img_width)
    clf.fit_generator(load_generator(train_data_dir, target_size, batch_size, train_gen_conf),
                      load_generator(val_data_dir, target_size, batch_size, val_gen_conf),
                      num_train=num_train,
                      num_val=num_test,
                      num_epoch=args.epoch)

    if args.save_file is not None:
        clf.save(args.save_file)
