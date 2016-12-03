#!/usr/bin/env python
import argparse

from cnn_feature import CustomMLPClassifier, classifiers

__version__ = "0.1"


def main(args):
    if args.goal == 'extract':
        from cnn_feature import CNNFeatureExtractor
        from preprocess_utils import create_from_yaml

        CNNFeatureExtractor().extract_feature(
            data_gen=create_from_yaml(args.gen_conf),
            feature_file=args.feature_file,
            architecture=args.architecture,
            nb_factor=args.factor)

    elif args.goal == 'viz':
        from cnn_feature import CNNFeatureExtractor
        CNNFeatureExtractor().visualize_intermediate(args.image_file, args.output_dir, architecture=args.architecture,
                                                     target_size=(args.image_height, args.image_width))
    elif args.goal == 'classify':
        from cnn_feature import ClassifierPool, CNNFeatureExtractor
        X_train, y_train, train_classes = CNNFeatureExtractor.load_features(feature_file=args.train_feature)
        reverse = dict(zip(train_classes.values(), train_classes.keys()))
        X_test, y_test, test_classes = CNNFeatureExtractor.load_features(feature_file=args.test_feature)
        y_test_new = [reverse[test_classes[label]] for label in y_test]
        print("Training has %d species, test has %d species" % (len(train_classes), len(test_classes)))
        clf = ClassifierPool(classifier_name=args.classifier, nb_features=X_train.shape[1])
        clf.train_and_score(X_train, y_train, X_test, y_test_new, test_class=test_classes,
                            model_file=args.model_file, results_file=args.result_file)

        # clf.fit(X_train, y_train)
        # if args.model_file is not None:
        #     clf.save(args.model_file)

    elif args.goal == 'cnn_classify':
        from cnn import run_cnn_classify
        run_cnn_classify(args)

    elif args.goal == 'split':
        from preprocess_utils import split_images
        split_images(args.data_dir, args.train_size, args.test_size)

    elif args.goal == 'mlp':
        from preprocess_utils import split_images
        from cnn_feature import ClassifierPool, CNNFeatureExtractor
        X_train, y_train, train_classes = CNNFeatureExtractor.load_features(feature_file=args.feature_file)
        CustomMLPClassifier()\
            .fit(X_train, y_train, batch_size=args.batch_size, nb_epoch=args.epoch)\
            .save(args.save_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-V', '--version', action='version', version=__version__)

    # ================================================
    subparsers = parser.add_subparsers(dest='goal')
    # ------------------------------------------------
    # extract
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument('-x', '--factor', default=1, help="factor for data augumentation")
    extract_parser.add_argument('-f', '--feature_file', required=True, help="the feature file to save to")
    extract_parser.add_argument('gen_conf', help="the path to the generator config")
    extract_parser.add_argument('-a', '--architecture', default='vgg16', choices=['vgg16', 'vgg19', 'resnet50'],
                                help="the CNN architecture to use")
    # ------------------------------------------------
    # viz
    viz_parser = subparsers.add_parser("viz")
    viz_parser.add_argument('-f', '--output_dir', required=True, help="the feature file to save to")
    viz_parser.add_argument('gen_conf', help="the path to the config")
    viz_parser.add_argument('image_file', help="the path to the image")
    viz_parser.add_argument('-a', '--architecture', default='vgg16', choices=['vgg16', 'vgg19', 'resnet50'],
                            help="the CNN architecture to use")
    # ------------------------------------------------
    # classify
    classify_parser = subparsers.add_parser("classify")
    classify_parser.add_argument('-c', '--classifier', default='SVC', choices=classifiers.keys(), help="classifier")
    classify_parser.add_argument('-f', '--train_feature', help="train feature file to load from")
    classify_parser.add_argument('-t', '--test_feature', help="test feature file to load from")
    classify_parser.add_argument('-m', '--model_file', help="file to save model and weight")
    classify_parser.add_argument('-r', '--result_file', help="file to write test results")
    # ------------------------------------------------
    cnn_parser = subparsers.add_parser('cnn_classify')
    cnn_parser.add_argument('-e', '--epoch', type=int, default=50, help="the number of epochs to run")
    cnn_parser.add_argument('-s', '--save_file', help="the file that the weight are saved to")
    cnn_parser.add_argument('gen_conf', help="the path to the config")
    # ------------------------------------------------
    split_parser = subparsers.add_parser('split', description='split images in a folder to train and val')
    split_parser.add_argument('--train_size', help="num of samples or proportion of samples for train")
    split_parser.add_argument('--test_size', help="num of samples or proportion of samples for validation")
    split_parser.add_argument('data_dir', help="the data dir with a images folder")
    # ------------------------------------------------
    top_cnn_parser = subparsers.add_parser('mlp', description='train top layer with pre-trained weights')
    top_cnn_parser.add_argument('--batch_size', type=int, default=8,  help="batch size")
    top_cnn_parser.add_argument('-e',  '--epoch', type=int, default=10, help="the number of epochs to run")
    top_cnn_parser.add_argument('-f', '--feature_file', help="the feature file to load from")
    top_cnn_parser.add_argument('-s', '--save_file', help="the file that the weight are saved to")
    # ================================================

    main(parser.parse_args())
