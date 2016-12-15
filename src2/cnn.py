from datetime import datetime

from keras.callbacks import EarlyStopping
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Dropout, Flatten, Dense, AveragePooling2D
from keras.models import Sequential


class CNNClassifier(object):
    def __init__(self, num_classes=2, target_size=(256, 256)):
        # dimensions of our images.
        self.img_height, self.img_width = target_size

        self.num_classes = num_classes
        self.model = self.create_model()
        self.model.summary()

    def create_model(self):
        loss = 'categorical_crossentropy'
        output_dim = self.num_classes
        final_activation = 'softmax'

        print((3, self.img_height, self.img_width))

        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(self.img_height, self.img_width, 3)))
        model.add(Convolution2D(32, 3, 3, activation='relu', name='conv0'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2'))
        model.add(AveragePooling2D(pool_size=(28, 28)))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim, activation=final_activation))

        model.compile(loss=loss,
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        return model

    def fit_generator(self, train_generator, val_generator, num_train, num_val, num_epoch=50):
        callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
        self.model.fit_generator(
            train_generator,
            samples_per_epoch=num_train,
            nb_epoch=num_epoch,
            validation_data=val_generator,
            nb_val_samples=num_val,
            callbacks=callbacks)

    def save(self, h5_file):
        self.model.save_weights(h5_file)


def run_cnn_classify(conf):

    train_gen = conf.train_gen
    val_gen = conf.test_gen

    clf = CNNClassifier(num_classes=train_gen.nb_class, target_size=train_gen.target_size)
    clf.fit_generator(train_gen, val_gen, num_train=train_gen.nb_sample * conf.factor, num_val=val_gen.nb_sample,
                      num_epoch=conf.epoch)

    if conf.weight_file is not None:
        clf.save(conf.weight_file)
    else:
        filename = conf.result_file.rsplit('/', 1)[-1]
        name = filename.rsplit('.', 1)[0]
        dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = 'history/%s_%s.h5' % (name, dt)
        clf.save(save_file)
