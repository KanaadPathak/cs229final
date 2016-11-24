import shutil
import os
import argparse


def split(data_dir, test_size=None, train_size=None):
    for a_class in os.listdir('%s/images/' % data_dir):
        class_dir = '%s/images/%s' % (data_dir, a_class)
        if not os.path.isdir(class_dir):
            # non-dir is not a class folder
            continue
        print('splitting class: %s' % a_class)
        train_class_dir = '%s/train/%s' % (data_dir, a_class)
        val_class_dir = '%s/validation/%s' % (data_dir, a_class)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        sample_list = os.listdir(class_dir)

        train_bound = int(len(sample_list) * .75)
        if train_size is not None:
            if isinstance(train_size, float):
                train_bound = int(len(sample_list) * train_size)
            elif isinstance(train_size, int):
                train_bound = train_size

        test_bound = len(sample_list)
        if train_size is not None:
            if isinstance(test_size, float):
                test_bound = train_bound + int(len(sample_list) * test_size)
            elif isinstance(test_size, int):
                test_bound = train_bound + test_size

        count = 0
        for f in sample_list:
            supported_ext = ('.jpeg', '.jpg', '.png')
            if f.lower().endswith(supported_ext):
                if count < train_bound:
                    shutil.copy('%s/%s' % (class_dir, f),
                                '%s/%s' % (train_class_dir, f))
                elif count < test_bound:
                    shutil.copy('%s/%s' % (class_dir, f),
                                '%s/%s' % (val_class_dir, f))
                count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_size', help="num of samples or proportion of samples for train")
    parser.add_argument('--test_size', help="num of samples or proportion of samples for validation")
    parser.add_argument('data_dir', help="the data dir with a images folder")
    args = parser.parse_args()
    split(args.data_dir, args.train_size, args.test_size)
