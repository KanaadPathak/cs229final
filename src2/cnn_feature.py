from functools import reduce

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from resnet50 import ResNet50
from vgg16 import VGG16
from vgg19 import VGG19

from preprocess_utils import GeneratorLoader
from tqdm import tqdm
from operator import mul
import tables
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif


class ClassifierPool(object):
    def __init__(self):
        self.classifiers = [
            ('KNN', KNeighborsClassifier(10)),
            ('Linear SVM', SVC(kernel="linear", C=0.01)),
            ('Linear SVM2', LinearSVC(C=0.01)),
            ('RBF SVM',    SVC(C=0.1, probability=True)),
            # ('Nu SVM', NuSVC(probability=True)),
            # ('Gaussian Process', GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)),
            # ('Decision Tree', DecisionTreeClassifier()),
            # ('Random Forest', RandomForestClassifier()),
            # ('AdaBoost', AdaBoostClassifier()),
            # ('GradientBoost', GradientBoostingClassifier()),
            ('Neural Network', MLPClassifier()),
            ('Naive Bayes', GaussianNB()),
            ('LDA', LinearDiscriminantAnalysis()),
            ('QDA', QuadraticDiscriminantAnalysis())]

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

        log_cols = ["Classifier", "Accuracy"]
        log = pd.DataFrame(columns=log_cols)

        for name, clf in self.classifiers:
            print("=" * 30)
            print(name, )

            score = cross_val_score(clf, X, y).min()
            print("Accuracy Score: %.4f" % score)

            log_entry = pd.DataFrame([[name, score]], columns=log_cols)
            log = log.append(log_entry)

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