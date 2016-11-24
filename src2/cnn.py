from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from argparse import ArgumentParser
from yaml import load

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
        model.add(Dense(self.num_classes))
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
        

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--epoch', type=int, default=50, help="the number of epochs to run")
    parser.add_argument('-s', '--save_file', help="the file that the weight are saved to")
    parser.add_argument('config_file', help="the path to the config")
    args = parser.parse_args()

    with open(args.config_file, 'r') as stream:
        conf = load(stream)

    class_mode = 'binary'
    if conf['num_classes']> 2:
        class_mode = 'categorical'

    train_generator = ImageDataGenerator().flow_from_directory(
        conf['train_data']['dir'],
        target_size=(conf['img_height'], conf['img_width']),
        batch_size=conf['batch_size'],
        class_mode=class_mode)

    validation_generator = ImageDataGenerator().flow_from_directory(
        conf['val_data']['dir'],
        target_size=(conf['img_height'], conf['img_width']),
        batch_size=conf['batch_size'],
        class_mode=class_mode)

    clf = CNNClassifier(num_classes=conf['num_classes'], img_height=conf['img_height'], img_width=conf['img_width'])
    clf.fit_generator(train_generator, validation_generator, num_train=conf['train_data']['num'],
                      num_val=conf['val_data']['num'], num_epoch=args.epoch)

    if args.save_file is not None:
        clf.save(args.save_file)
