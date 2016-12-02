from keras.layers import Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing import image
from tqdm import trange

from resnet50 import ResNet50
from vgg16 import VGG16
from vgg19 import VGG19
from imagenet_utils import preprocess_input

import cv2
import tables
import numpy as np
import pandas as pd

from preprocess_utils import GeneratorLoader
from tqdm import tqdm
from operator import mul
from functools import reduce
import os

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split, GridSearchCV, KFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression


class ClassifierPool(object):
    def __init__(self):
        self.classifiers = [
            ('KNN', KNeighborsClassifier(), {'n_neighbors': [5, 10]})
            , ('Linear SVM', SVC(), {'kernel': ["linear", 'rbf'], 'C': [0.01, 0.025]})
            # , ('Nu SVM', NuSVC(probability=True), {})
            # , ('Gaussian Process', GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True))
            # , ('Decision Tree', DecisionTreeClassifier(), {})
            # , ('Random Forest', RandomForestClassifier(), {})
            # , ('AdaBoost', AdaBoostClassifier(), {})
            # , ('GradientBoost', GradientBoostingClassifier(), {})
            , ('Neural Network', MLPClassifier(), {})
            , ('Naive Bayes', GaussianNB(), {})
            , ('LDA', LinearDiscriminantAnalysis(), {})
            # ,('QDA', QuadraticDiscriminantAnalysis(), {})
        ]

    def feature_selection(self, X, y):
        X = VarianceThreshold(threshold=(.9 * (1 - .9))).fit_transform(X, y)
        X = SelectKBest(f_classif, k=min(X.shape[1], 3000)).fit_transform(X, y)

        return X

    def scale(self, X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    def classify(self, X, y):
        X = self.feature_selection(X, y)
        X = self.scale(X)
        print(X.shape)

        # log_cols = ["Classifier", "Accuracy"]
        # log = pd.DataFrame(columns=log_cols)

        for name, clf, param_grid in self.classifiers:
            print("=" * 30)
            print(name, )
            # cv = KFold(2)
            clf = GridSearchCV(clf, param_grid, n_jobs=1)
            # clf.fit(X, y)
            # print('Training Accuracy %.4f with params: %s' % (clf.best_score_, clf.best_params_))

            score = cross_val_score(clf, X, y).min()
            print("Accuracy Score: %.4f" % score)

            # log_entry = pd.DataFrame([[name, score]], columns=log_cols)
            # log = log.append(log_entry)

        print("=" * 30)


class CNNFeatureExtractor(object):
    def __init__(self):
        pass

    @staticmethod
    def select_architecture(architecture):
        if architecture == 'vgg16':
            return VGG16(weights='imagenet', include_top=False)
        elif architecture == 'vgg19':
            return VGG19(weights='imagenet', include_top=False)
        elif architecture == 'resnet50':
            return ResNet50(weights='imagenet', include_top=False)

    def extract_feature(self, data_dir, feature_file, architecture='vgg16', target_size=(256, 256), batch_size=8):
        model = self.select_architecture(architecture)

        data_gen = GeneratorLoader(target_size=target_size, batch_size=batch_size).load_generator(data_dir)

        with tqdm(total=data_gen.nb_sample) as pbar, \
                tables.open_file(feature_file, mode='w') as f:

            atom = tables.Float64Atom()
            cnn_input_shape = (batch_size, *data_gen.image_shape)
            cnn_output_shape = model.get_output_shape_for(cnn_input_shape)
            print(cnn_output_shape)
            single_sample_feature_shape = reduce(mul, cnn_output_shape[1:])
            feature_arr = f.create_earray(f.root, 'features', atom, (0, single_sample_feature_shape))
            label_arr = f.create_earray(f.root, 'labels', atom, (0, ))
            table_def = {
                'classid': tables.Int32Col(),
                'name': tables.StringCol(itemsize=60),
            }
            class_table = f.create_table(f.root, 'classes', table_def)
            row = class_table.row
            for name, classid in data_gen.class_indices.items():
                row['classid'] = classid
                row['name'] = name
                row.append()
            class_table.flush()

            for X, y in data_gen:
                features = model.predict(X).reshape((X.shape[0], single_sample_feature_shape))
                feature_arr.append(features)
                label_arr.append(y.argmax(1))
                pbar.update(X.shape[0])
                if pbar.n >= data_gen.nb_sample:
                    break

    @staticmethod
    def _convert(img):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return preprocess_input(x)

    def visualize_intermediate(self, img_path, output_dir, architecture='vgg16', target_size=(256, 256)):
        model = self.select_architecture(architecture)

        img = image.load_img(img_path, target_size=target_size)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        x = self._convert(img)

        middle_layers = [layer for layer in model.layers if isinstance(layer, Convolution2D)]
        get_features = K.function([model.layers[0].input, K.learning_phase()], [l.output for l in middle_layers])
        # we only have one sample of dim: (height, width, features)
        all_features = [f[0].transpose(2, 0, 1) for f in get_features([x, 1])]
        all_names = [l.name for l in middle_layers]
        print('all layers: %s' % ', '.join(all_names))

        with tqdm(total=sum(l.shape[0] for l in all_features)) as pbar:
            for layer_name, features_of_layer in zip(all_names, all_features):
                dir_path = os.path.join(output_dir, img_name, layer_name)
                os.makedirs(dir_path, exist_ok=True)
                for j in range(features_of_layer.shape[0]):
                    feat_normalized = self._normalize(features_of_layer[j])
                    dims = '_'.join(map(lambda i: str(i), feat_normalized.shape))
                    output_path = os.path.join(dir_path, 'feature_%s_%s.jpg' % (j, dims))
                    im_color = cv2.applyColorMap(feat_normalized, cv2.COLORMAP_AUTUMN)
                    cv2.imwrite(output_path, im_color)
                    pbar.update(1)

    @staticmethod
    def _normalize(img):
        img_max = img.max()
        img_min = img.min()
        img -= img_min
        if img_max != img_min:
            img *= 255 / (img_max - img_min)
        return np.uint8(img)

    def load_features(self, feature_file):
        with tables.open_file(feature_file, mode='r') as f:
            X = f.root.features[:, :].astype(float)
            y = f.root.labels[:].astype(int)
            classes = dict((int(r['classid']), r['name']) for r in f.root.classes.iterrows())
        return X, y, classes

    def train_top_model(self, feature_file, weight_file=None, batch_size=32, nb_epoch=10):
        X, y, classes = self.load_features(feature_file)
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=.7, stratify=y)
        y_train = np_utils.to_categorical(y_train)
        y_val = np_utils.to_categorical(y_val)

        loss = 'binary_crossentropy'
        output_dim = 1
        final_activation = 'sigmoid'
        if len(classes) > 2:
            loss = 'categorical_crossentropy'
            output_dim = len(classes)
            final_activation = 'softmax'

        model = Sequential()
        model.add(Dense(4096, activation='relu', input_dim=X_train.shape[1]))
        model.add(Dense(4096, activation='relu', name='fc2'))
        model.add(Dense(output_dim, activation=final_activation, name='predictions'))

        print(loss, output_dim, final_activation)

        model.compile(optimizer='rmsprop', loss=loss, metrics=['accuracy'])

        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_val, y_val))

        if weight_file is not None:
            model.save_weights(weight_file)