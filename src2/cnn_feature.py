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
import time

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
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, f_classif, f_regression, RFECV
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib


class ClassifierPool(object):
    def __init__(self):
        self.classifiers = [
            #('KNN', KNeighborsClassifier(), {'n_neighbors': [5, 10]})
             ('Linear SVM', SVC(), {'kernel': ["linear"], 'C': np.logspace(-2, -1, 1, endpoint=True)})
            #('RBF SVM', SVC(), {'kernel': ['rbf'], 'C': np.logspace(-2, 1, 4, endpoint=True),
            #                    'gamma': np.logspace(-1,3,5, endpoint=True)})
            # , ('Nu SVM', NuSVC(probability=True), {})
            # , ('Gaussian Process', GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True))
            # , ('Decision Tree', DecisionTreeClassifier(), {})
            # , ('Random Forest', RandomForestClassifier(), {})
            # , ('AdaBoost', AdaBoostClassifier(), {})
            # , ('GradientBoost', GradientBoostingClassifier(), {})
            #, ('Neural Network', MLPClassifier(), {})
            #, ('Naive Bayes', GaussianNB(), {})
            #, ('LDA', LinearDiscriminantAnalysis(), {})
            # ,('QDA', QuadraticDiscriminantAnalysis(), {})
        ]
        self.selector1 = None
        self.selector2 = None
        self.scaler = None

    def feature_selection(self, X_train, y_train, X_test):
        t0 = time.time()
        print("before selection")
        print(X_train.shape)

        print("fitting first selector")
        self.selector1= VarianceThreshold(threshold=(.9 * (1 - .9)))
        X_train = self.selector1.fit_transform(X_train, y_train)
        #print("fitting second selector")
        #self.selector2 = SelectKBest(mutual_info_classif, k=min(X_train.shape[1], 500))
        #X_train = self.selector2.fit_transform(X_train, y_train)
        print("transforming on test")

        #backword search
        #svc = SVC(kernel="linear", C=0.001)
        #rfecv = RFECV(estimator=svc, step=10, cv=StratifiedKFold(3), n_jobs=-1, scoring='accuracy', verbose=9)
        #X_train = rfecv.fit_transform(X_train, y_train)
        #print("Backward search gives number of features : %d" % rfecv.n_features_)
        #print("before selection")
        #print(X_train.shape)

        X_test = self.selector1.transform(X_test)
        #X_test = self.selector2.transform(X_test)
        #X_test = rfecv.predict(X_test)

        print("after selection: %.2f" % (time.time() - t0))
        print(X_train.shape)
        print(X_test.shape)

        return (X_train, X_test)

    def scale(self, X_train, X_test):
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        print("after scaling")
        print(X_train.shape)
        return (X_train, X_test)

    def pca(self, X_train, X_test, scaler):
        #dimension reduction + whitening
        pca = PCA(n_components=min(1000, X_train.shape[1]), whiten=True)
        X_train = pca.fit_transform(X_train)
        print("after PCA")
        print(X_train.shape)
        #normalize against, per suggested in the paper
        X_train = scaler.fit_transform(X_train)

        #feature scaling
        X_test = pca.transform(X_test)
        #normalize after PCA
        X_test = scaler.transform(X_test)
        return (X_train, X_test)

    def classify(self, X_train, y_train, X_test, y_test, test_class=None, results_file=None, load_clf=None):
        if load_clf is not None:
            print("loading clf from %s"%(load_clf))
            d = joblib.load(load_clf)
            clf = d['clf']
            best_clf = clf
            if 'selector1' in d:
                self.selector1 = d['selector1']
            if 'selector2' in d:
                self.selector2 = d['selector2']
            if 'scaler' in d:
                self.scaler = d['scaler']
            test_class = d['y_class']

            if self.scaler is not None:
                X_test = self.scaler.transform(X_test)
            if self.selector1 is not None:
                X_test = self.selector1.transform(X_test)
            if self.selector2 is not None:
                X_test = self.selector2.transform(X_test)
            best_predict = clf.predict(X_test)
            best_score = clf.score(X_test, y_test)
            print("%s Test Accuracy: %0.4f" % (str(clf), best_score))
        else:
            #normalizatoin first!
            (X_train, X_test) = self.scale(X_train, X_test)

            #X_train, X_test = self.pca(X_train, X_test, scaler)
            #feature selection
            (X_train, X_test) = self.feature_selection(X_train, y_train, X_test)
            best_score = 0.0; best_predict = []; best_clf = None
            for name, clf, param_grid in self.classifiers:
                print("=" * 30)
                print(name, )
                # cv = KFold(2)
                clf = GridSearchCV(clf, param_grid, verbose=9, cv=KFold(n_splits=3), n_jobs=-1)
                clf.fit(X_train, y_train)
                print('Training Accuracy %.4f with params: %s' % (clf.best_score_, clf.best_params_))

                y_predict = clf.predict(X_test)
                score = clf.score(X_test, y_test)
                print("%s Test Accuracy: %0.4f" % (str(clf), score))
                if score > best_score:
                    best_score = score
                    best_predict = y_predict
                    best_clf = clf

        print("=" * 30)
        for (truth, predicted) in zip(y_test, best_predict):
            print("%s(%d) -> %s(%d)" % (test_class[truth], truth, test_class[predicted], predicted))


        if results_file is not None:
            d = {
                'scaler' : self.scaler,
                'selector1':self.selector1,
                'selector2':self.selector2,
                'clf':best_clf,
                'y_predict': best_predict,
                'y_test': y_test,
                'score': best_score,
                'y_class': test_class}
            joblib.dump(d, results_file, protocol=2)


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

    def extract_feature(self, data_dir, feature_file, architecture='vgg16', target_size=(256, 256), batch_size=8, aug=False):
        model = self.select_architecture(architecture)

        if aug:
            data_gen_args = dict(
                rotation_range=10,
                channel_shift_range=100.,
            )
            nb_factor = 5
            print("Image augementation is on, factor=%d"%(nb_factor))
        else:
            data_gen_args = None
            print("Image augementation is off")
            nb_factor = 1
        data_gen = GeneratorLoader(target_size=target_size, batch_size=batch_size, generator_params=data_gen_args).load_generator(data_dir)


        with tqdm(total=data_gen.nb_sample * nb_factor) as pbar, \
                tables.open_file(feature_file, mode='w') as f:

            atom = tables.Float64Atom()
            cnn_input_shape = (batch_size, *data_gen.image_shape)
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

            print("feature dimension: %d" % single_sample_feature_shape)
            for X, y in data_gen:
                features = model.predict(X).reshape((X.shape[0], single_sample_feature_shape))
                feature_arr.append(features)
                label_arr.append(y.argmax(1))
                pbar.update(X.shape[0])
                if pbar.n >= data_gen.nb_sample * nb_factor:
                    break

    def convert(self, img):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return preprocess_input(x)

    def visualize_intermediate(self, img_path, output_dir, architecture='vgg16', target_size=(256, 256)):
        model = self.select_architecture(architecture)

        img = image.load_img(img_path, target_size=target_size)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        x = self.convert(img)

        middle_layers = [layer for layer in model.layers if isinstance(layer, Convolution2D)]
        get_features = K.function([model.layers[0].input, K.learning_phase()], [l.output for l in middle_layers])
        # we only have one sample of dim: (height, width, features)
        all_features = [f[0].transpose(2, 0, 1) for f in get_features([x, 1])]
        all_names = [l.name for l in middle_layers]
        print(all_names)

        with tqdm(total=sum(l.shape[0] for l in all_features)) as pbar:
            for layer_name, features_of_layer in zip(all_names, all_features):
                dir_path = os.path.join(output_dir, img_name, layer_name)
                os.makedirs(dir_path, exist_ok=True)
                for j in range(features_of_layer.shape[0]):
                    output_path = os.path.join(dir_path, 'feature_%s.jpg' % j)
                    feat_normalized = self._normalize(features_of_layer[j])
                    im_color = cv2.applyColorMap(feat_normalized, cv2.COLORMAP_JET)
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
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=.8, stratify=y)
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
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu', name='fc2'))
        model.add(Dense(output_dim, activation=final_activation, name='predictions'))

        model.compile(optimizer='rmsprop', loss=loss, metrics=['accuracy'])

        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_val, y_val))

        if weight_file is not None:
            model.save_weights(weight_file)
