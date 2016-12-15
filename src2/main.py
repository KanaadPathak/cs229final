#!/usr/bin/env python
import argparse

__version__ = "0.1"

import logging


def main(args):
    if args.goal == 'extract':
        from cnn_feature import CNNFeatureExtractor
        from preprocess_utils import Configuration

        conf = Configuration(args.conf_file)

        extractor = CNNFeatureExtractor(architecture=conf.architecture, image_shape=conf.train_gen.image_shape)
        extractor.extract_feature(data_gen=conf.train_gen, feature_file=conf.train_feature, nb_factor=conf.factor)
        extractor.extract_feature(data_gen=conf.test_gen, feature_file=conf.test_feature)

    elif args.goal == 'viz':
        from cnn_feature import CNNFeatureExtractor
        from preprocess_utils import Configuration

        conf = Configuration(args.conf_file)

        extractor = CNNFeatureExtractor(architecture=conf.architecture, image_shape=conf.train_gen.image_shape)
        extractor.visualize_intermediate(args.image_file, args.output_dir, target_size=conf.target_size)

    elif args.goal == 'extract_any':
        from cnn_feature import CNNFeatureExtractor
        from preprocess_utils import Configuration

        conf = Configuration(args.conf_file)

        extractor = CNNFeatureExtractor(architecture='resnet50', image_shape=conf.train_gen.image_shape)
        extractor.extract_any(data_gen=conf.train_gen, feature_file=conf.train_feature, nb_factor=conf.factor)
        extractor.extract_any(data_gen=conf.test_gen, feature_file=conf.test_feature)

    elif args.goal == 'classify':
        from cnn_feature import ClassifierPool, CNNFeatureExtractor
        from preprocess_utils import Configuration

        conf = Configuration(args.conf_file)

        X_train, y_train, train_classes = CNNFeatureExtractor.load_features(feature_file=conf.train_feature)
        X_test, y_test, test_classes = CNNFeatureExtractor.load_features(feature_file=conf.test_feature)

        reverse = dict(zip(train_classes.values(), train_classes.keys()))
        y_test = [reverse[test_classes[label]] for label in y_test]
        print("Training has %d species, test has %d species" % (len(train_classes), len(test_classes)))
        clf = ClassifierPool(classifier_name=conf.classifier_name, nb_features=X_train.shape[1])
        clf.train_and_score(X_train, y_train, X_test, y_test, test_class=test_classes,
                            model_file=conf.model_file, results_file=conf.result_file)

        # clf.fit(X_train, y_train)
        # if train_conf.model_file is not None:
        #     clf.save(train_conf.model_file)
    elif args.goal == 'fine_tune':
        from cnn_feature import ClassifierPool, CNNFeatureExtractor
        from preprocess_utils import Configuration

        conf = Configuration(args.conf_file)

        CNNFeatureExtractor(
            architecture=conf.architecture,
            image_shape=conf.train_gen.image_shape
        ).fine_tune(
            train_gen=conf.train_gen,
            test_gen=conf.test_gen,
            feature_file=conf.train_feature,
            nb_epoch=conf.epoch,
            nb_factor=conf.factor)


    elif args.goal == 'augment':
        from cnn_feature import CNNFeatureExtractor
        from preprocess_utils import Configuration

        conf = Configuration(args.conf_file)

        CNNFeatureExtractor.augment(args.image_file, args.output_dir, conf.generator_params,
                                    batch_size=conf.batch_size, target_size=conf.target_size)

    elif args.goal == 'cnn_classify':
        from cnn import run_cnn_classify
        from preprocess_utils import Configuration

        conf = Configuration(args.conf_file)
        run_cnn_classify(conf)

    elif args.goal == 'split':
        from preprocess_utils import split_images
        split_images(args.data_dir, int(args.train_size), int(args.test_size))

    elif args.goal == 'mlp':
        from preprocess_utils import Configuration
        from cnn_feature import CNNFeatureExtractor, CustomMLPClassifier

        conf = Configuration(args.conf_file)

        X_train, y_train, train_classes = CNNFeatureExtractor.load_features(conf.train_feature)
        X_test, y_test, test_classes = CNNFeatureExtractor.load_features(conf.test_feature)

        reverse = dict(zip(train_classes.values(), train_classes.keys()))
        y_test = [reverse[test_classes[label]] for label in y_test]
        # y_test = [conf.train_gen.class_indices[test_classes[label]] for label in y_test]
        print("Training has %d species, test has %d species" % (len(train_classes), len(test_classes)))

        clf = CustomMLPClassifier()
        clf.fit(X_train, y_train, X_test, y_test, batch_size=conf.batch_size, nb_epoch=conf.epoch)
        # clf.save(conf.result_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-V', '--version', action='version', version=__version__)
    parser.add_argument('-c', '--conf_file', help="the path to the config file")

    # logging.basicConfig(filename='the.log', level=logging.DEBUG)

    # ================================================
    subparsers = parser.add_subparsers(dest='goal')
    # ------------------------------------------------
    # extract
    cur_parser = subparsers.add_parser("extract")
    # ------------------------------------------------
    # custom mlp
    cur_parser = subparsers.add_parser('mlp', description='train top layer with pre-trained weights')
    # ------------------------------------------------
    # classify
    cur_parser = subparsers.add_parser("classify")
    # ------------------------------------------------
    # classify
    cur_parser = subparsers.add_parser("fine_tune")
    # ------------------------------------------------
    # extract_any
    cur_parser = subparsers.add_parser("extract_any")
    # ------------------------------------------------
    # augment
    cur_parser = subparsers.add_parser("augment")
    cur_parser.add_argument('-o', '--output_dir', required=True, help="the feature file to save to")
    cur_parser.add_argument('image_file', help="the path to the image")
    # ------------------------------------------------
    # visualize middle layers
    cur_parser = subparsers.add_parser("viz")
    cur_parser.add_argument('-o', '--output_dir', required=True, help="the feature file to save to")
    cur_parser.add_argument('image_file', help="the path to the image")
    # ------------------------------------------------
    cnn_parser = subparsers.add_parser('cnn_classify')
    # ------------------------------------------------
    cur_parser = subparsers.add_parser('split', description='split images in a folder to train and val')
    cur_parser.add_argument('--train_size', help="num of samples or proportion of samples for train")
    cur_parser.add_argument('--test_size', help="num of samples or proportion of samples for test")
    cur_parser.add_argument('data_dir', help="the data dir with a images folder")
    # ================================================

    main(parser.parse_args())
