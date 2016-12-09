import os
from functools import reduce
from operator import mul

import cv2
import numpy as np
import tables
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.regularizers import l2, activity_l2
from keras.utils import np_utils
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from imagenet_utils import preprocess_input
from resnet50 import ResNet50
from vgg16 import VGG16
from vgg19 import VGG19

classifiers = {
    'SVC': (SVC(), {'kernel': ["linear"], 'C': [0.01, 0.1, 1.0]}),
    'KNN': (KNeighborsClassifier(), {'n_neighbors': [5, 10]}),
    'GaussianProcess': (GaussianProcessClassifier(), {'kernel': 1.0 * RBF(1.0), 'warm_start': True}),
    'DecisionTree': (DecisionTreeClassifier(), {}),
    'RandomForest': (RandomForestClassifier(), {}),
    'AdaBoost': (AdaBoostClassifier(), {}),
    'GradientBoosting': (GradientBoostingClassifier(), {}),
    'MLP': (MLPClassifier(), {}),
    'NaiveBayes': (GaussianNB(), {}),
    'LDA': (LinearDiscriminantAnalysis(), {}),
}


# noinspection PyPep8Naming
class ClassifierPool(object):
    def __init__(self, classifier_name='SVC', nb_features=1000):
        self.classifier_name = classifier_name
        self.classifier, self.param_grid = classifiers[classifier_name]

        self.preprocessors = [
            StandardScaler(),
            # feature scaling
            # PCA(n_components=min(256, nb_features), whiten=True),
            # normalize against after pca, per suggested in the paper
            # StandardScaler(),
            # feature selection
            # VarianceThreshold(threshold=(.9 * (1 - .9))),
            # SelectKBest(mutual_info_classif, k=min(nb_features, 1000)),
        ]

    def save(self, model_file):
        joblib.dump(self.__dict__, model_file)

    def load(self, model_file):
        self.__dict__.update(joblib.load(model_file))

    def fit(self, X, y):
        print('initial shape:', X.shape)
        for estimator in self.preprocessors:
            try:
                estimator.k = min(estimator.k, X.shape[1])
            except AttributeError:
                pass
            X = estimator.fit_transform(X, y)
            print("after %s:" % estimator.__class__.__name__, X.shape)

        print("=" * 30)
        print('using classifier: %s' % self.classifier_name, )
        cv = StratifiedKFold(n_splits=3)
        clf = GridSearchCV(self.classifier, self.param_grid, verbose=9, cv=cv, n_jobs=-1)
        clf.fit(X, y)
        self.classifier = clf.best_estimator_
        print('Training Accuracy %.4f with params: %s' % (clf.best_score_, clf.best_params_))

    def transform(self, X, y=None):
        print('initial shape:', X.shape)
        for estimator in self.preprocessors:
            X = estimator.transform(X)
            print("after %s:" % estimator.__class__.__name__, X.shape)
        return X

    def predict(self, X):
        return self.classifier.predict(X)

    def train_and_score(self, X_train, y_train, X_test, y_test, test_class=None, model_file=None, results_file=None):
        self.fit(X_train, y_train)

        if model_file is not None:
            self.save(model_file)
        print("%s model saved" % self.classifier_name)
        print('Testing flavia')
        X_test = self.transform(X_test)
        y_predict = self.predict(X_test)
        score = self.classifier.score(X_test, y_test)

        print("%s Test Accuracy: %0.4f" % (self.classifier_name, score))

        if results_file is not None:
            d = {'y_predict': y_predict, 'y_test': y_test, 'score': score, 'y_class': test_class}
            joblib.dump(d, results_file)


# noinspection PyPep8Naming
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

    def extract_feature(self, data_gen, feature_file, architecture='vgg16', nb_factor=1):
        model = self.select_architecture(architecture)

        with tqdm(total=data_gen.nb_sample * nb_factor) as pbar, \
                tables.open_file(feature_file, mode='w') as f:

            atom = tables.Float64Atom()
            cnn_input_shape = (data_gen.batch_size, *data_gen.image_shape)
            cnn_output_shape = model.get_output_shape_for(cnn_input_shape)
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
                if pbar.n >= data_gen.nb_sample * nb_factor:
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

    @staticmethod
    def load_features(feature_file):
        with tables.open_file(feature_file, mode='r') as f:
            X = f.root.features[:, :].astype(float)
            y = f.root.labels[:].astype(int)
            classes = dict((int(r['classid']), r['name']) for r in f.root.classes.iterrows())
        return X, y, classes


class CustomMLPClassifier(ClassifierMixin):
    def __init__(self, batch_size=32, nb_epoch=10):
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.model = Sequential()

    @staticmethod
    def _create_model(nb_classes, nb_features):
        loss = 'categorical_crossentropy'
        output_dim = nb_classes
        final_activation = 'softmax'

        print(loss, output_dim, final_activation)

        model = Sequential()
        model.add(Dense(1024, activation='relu', input_dim=nb_features))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(output_dim, activation=final_activation, name='predictions'))
        optimizer = SGD(lr=1e-4, momentum=0.9)
        # optimizer = 'rmsprop'
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        return model

    def predict(self, X):
        return self.model.predict(X, self.batch_size)

    def fit(self, X_train, y_train, X_val, y_val, batch_size=32, nb_epoch=10):
        nb_classes = len(np.unique(y_train))
        nb_features = X_train.shape[1]

        # PCA(n_components=min(1000, nb_features), whiten=True)

        # X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=.7, stratify=y)
        y_train = np_utils.to_categorical(y_train, nb_classes=nb_classes)
        y_val = np_utils.to_categorical(y_val, nb_classes=nb_classes)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        self.model = self._create_model(nb_classes, nb_features)
        hist = self.model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_val, y_val),
                       callbacks=[early_stopping])
        print(hist.history)

    def load(self, model_file):
        self.model = load_model(model_file)

    def save(self, model_file):
        self.model.save(model_file)
