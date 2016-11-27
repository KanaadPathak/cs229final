from functools import reduce

from preprocess_utils import load_generator
from vgg16 import VGG16
from tqdm import tqdm
from operator import mul
import tables

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
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, chi2, f_classif
import pandas as pd


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
        selectors = [
            VarianceThreshold(threshold=(.9 * (1 - .9))),
            SelectKBest(f_classif, k=3000)
            # SelectPercentile(f_classif, percentile=20)
        ]

        for sel in selectors:
            X = sel.fit_transform(X, y)
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

    def extract_feature(self, data_dir, feature_file, target_size=(256, 256), batch_size=8):
        model = VGG16(weights='imagenet', include_top=False)

        data_gen = load_generator(data_dir, target_size=target_size, batch_size=batch_size)

        with tqdm(total=data_gen.nb_sample) as pbar, tables.open_file(feature_file, mode='w') as f:
            atom = tables.Float64Atom()
            cnn_input_shape = (batch_size, *data_gen.image_shape)
            cnn_output_shape = model.get_output_shape_for(cnn_input_shape)
            single_sample_feature_shape = reduce(mul, cnn_output_shape[1:])
            feature_arr = f.create_earray(f.root, 'features', atom, (0, single_sample_feature_shape))
            label_arr = f.create_earray(f.root, 'labels', atom, (0, ))

            for X, y in data_gen:
                batch_samples = X.shape[0]
                features = model.predict(X).reshape((batch_samples, single_sample_feature_shape))
                feature_arr.append(features)
                label_arr.append(y.argmax(1))
                pbar.update(batch_samples)
                if pbar.n >= data_gen.nb_sample:
                    break

    def load_features(self, feature_file):
        with tables.open_file(feature_file, mode='r') as f:
            X = f.root.features[:, :]
            y = f.root.labels[:]
        return X, y
