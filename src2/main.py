#!/usr/bin/env python
import argparse

__version__ = "0.1"


def main(args):
    from cnn_feature import ClassifierPool, CNNFeatureExtractor
    from cnn import run_cnn_classify

    if args.goal == 'extract':
        CNNFeatureExtractor().extract_feature(args.data_dir, args.feature_file)
    elif args.goal == 'classify':
        X, y = CNNFeatureExtractor().load_features(feature_file=args.feature_file)
        ClassifierPool().classify(X, y)
    elif args.goal == 'cnn_classify':
        run_cnn_classify(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-V', '--version', action='version', version=__version__)

    # ================================================
    subparsers = parser.add_subparsers(dest='goal')
    # ------------------------------------------------
    # extract
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument('data_dir', help="the path to the config")
    extract_parser.add_argument('-f', '--feature_file', required=True, help="the file that the weight are saved to")
    # ------------------------------------------------
    # classify
    classify_parser = subparsers.add_parser("classify")
    classify_parser.add_argument('-f', '--feature_file', required=True, help="the file that the weight are load from")
    # ------------------------------------------------
    cnn_parser = subparsers.add_parser('cnn_classify')
    cnn_parser.add_argument('-e', '--epoch', type=int, default=50, help="the number of epochs to run")
    cnn_parser.add_argument('-s', '--save_file', help="the file that the weight are saved to")
    cnn_parser.add_argument('config_file', help="the path to the config")
    # ================================================

    main(parser.parse_args())
