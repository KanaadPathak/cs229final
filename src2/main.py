#!/usr/bin/env python
import argparse
__version__ = "0.1"


def main(args):
    if args.goal == 'extract':
        from cnn_feature import CNNFeatureExtractor
        CNNFeatureExtractor().extract_feature(args.data_dir, args.feature_file, architecture=args.architecture,
                                              target_size=(args.image_height, args.image_width),
                                              batch_size=args.batch_size, aug=args.aug)
    elif args.goal == 'viz':
        from cnn_feature import CNNFeatureExtractor
        CNNFeatureExtractor().visualize_intermediate(args.image_file, args.output_dir, architecture=args.architecture,
                                                     target_size=(args.image_height, args.image_width))
    elif args.goal == 'classify':
        from cnn_feature import ClassifierPool, CNNFeatureExtractor
        X_train, y_train, train_classes = CNNFeatureExtractor().load_features(feature_file=args.feature_file)
        reverse = dict(zip( train_classes.values(), train_classes.keys()))
        X_test, y_test, test_classes = CNNFeatureExtractor().load_features(feature_file=args.test_feature)
        y_test_new = []; test_classes_new = {}
        test_cnt = {}
        for label in y_test:
            species_name = test_classes[label]
            new_label = reverse[species_name]
            y_test_new.append(new_label)
            if new_label not in test_classes_new:
                test_classes_new[new_label] = species_name
            if new_label not in test_cnt:
                test_cnt[new_label] = 1
            else:
                test_cnt[new_label] += 1
        print("Training has %d species, test has %d species"%(len(train_classes), len(test_classes_new)))
        if args.print_test:
            for (k, cnt) in test_cnt.items():
                print("%s(%d) has samples=%d" % (train_classes[k], k, cnt))
        ClassifierPool().classify(X_train, y_train, X_test, y_test_new, train_classes, args.results, args.load_clf)


    elif args.goal == 'cnn_classify':
        from cnn import run_cnn_classify
        run_cnn_classify(args)

    elif args.goal == 'split':
        from preprocess_utils import split_images
        split_images(args.data_dir, args.train_size, args.test_size)

    elif args.goal == 'top_cnn_classify':
        from preprocess_utils import split_images
        from cnn_feature import ClassifierPool, CNNFeatureExtractor
        CNNFeatureExtractor().train_top_model(feature_file=args.feature_file, weight_file=args.save_file,
                                              batch_size=args.batch_size, nb_epoch=args.epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-V', '--version', action='version', version=__version__)

    # ================================================
    subparsers = parser.add_subparsers(dest='goal')
    # ------------------------------------------------
    # extract
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument('-f', '--feature_file', required=True, help="the feature file to save to")
    extract_parser.add_argument('-W', '--image_width', type=int, default=255, help="the target image width")
    extract_parser.add_argument('-H', '--image_height', type=int, default=255, help="the target image height")
    extract_parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    extract_parser.add_argument('--aug', action='store_true', default=False, help="turn on image augmentation")
    extract_parser.add_argument('data_dir', help="the path to the config")
    extract_parser.add_argument('-a', '--architecture', default='vgg16', choices=['vgg16', 'vgg19', 'resnet50', 'resnet48'],
                                help="the CNN architecture to use")
    # ------------------------------------------------
    # viz
    viz_parser = subparsers.add_parser("viz")
    viz_parser.add_argument('-f', '--output_dir', required=True, help="the feature file to save to")
    viz_parser.add_argument('-W', '--image_width', type=int, default=255, help="the target image width")
    viz_parser.add_argument('-H', '--image_height', type=int, default=255, help="the target image height")
    viz_parser.add_argument('image_file', help="the path to the image")
    viz_parser.add_argument('-a', '--architecture', default='vgg16', choices=['vgg16', 'vgg19', 'resnet50'],
                            help="the CNN architecture to use")
    # ------------------------------------------------
    # classify
    classify_parser = subparsers.add_parser("classify")
    classify_parser.add_argument('-f', '--feature_file', required=True, help="train feature file to load from")
    classify_parser.add_argument('-t', '--test_feature', required=True, help="test feature file to load from")
    classify_parser.add_argument('-r', '--results', help="file to write test results")
    classify_parser.add_argument('--load_clf', help="load input clf")
    classify_parser.add_argument('--print_test', action='store_true', help="give more info on test")
    # ------------------------------------------------
    cnn_parser = subparsers.add_parser('cnn_classify')
    cnn_parser.add_argument('-e', '--epoch', type=int, default=50, help="the number of epochs to run")
    cnn_parser.add_argument('-s', '--save_file', help="the file that the weight are saved to")
    cnn_parser.add_argument('config_file', help="the path to the config")
    # ------------------------------------------------
    split_parser = subparsers.add_parser('split', description='split images in a folder to train and val')
    split_parser.add_argument('--train_size', help="num of samples or proportion of samples for train")
    split_parser.add_argument('--test_size', help="num of samples or proportion of samples for validation")
    split_parser.add_argument('data_dir', help="the data dir with a images folder")
    # ------------------------------------------------
    top_cnn_parser = subparsers.add_parser('top_cnn_classify', description='train top layer with pre-trained weights')
    top_cnn_parser.add_argument('--batch_size', type=int, default=32,  help="batch size")
    top_cnn_parser.add_argument('-e',  '--epoch', type=int, default=10, help="the number of epochs to run")
    top_cnn_parser.add_argument('-f', '--feature_file', help="the feature file to load from")
    top_cnn_parser.add_argument('-s', '--save_file', help="the file that the weight are saved to")
    # ================================================

    main(parser.parse_args())
