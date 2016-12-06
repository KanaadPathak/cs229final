import os
import shutil

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd


def get_classes(data_dir):
    classes_ = []
    for a_class in os.listdir(data_dir):
        class_dir = '%s/%s' % (data_dir, a_class)
        if os.path.isdir(class_dir):
            classes_.append(a_class)
    return classes_


def count_image(data_dir):
    supported_ext = ('.jpeg', '.jpg', '.png')
    image_count = 0
    for root, dirs, files in os.walk(data_dir):
        image_count += sum(f.endswith(supported_ext) for f in files)
    return image_count


def create_from_dict(conf):
    batch_size = conf.get('batch_size', 32)
    target_size = (conf.get('img_height', 256), conf.get('img_width', 256))
    generator_params = conf.get('gen_params', {})
    data_dir = conf.get('data_dir')
    return GeneratorLoader(
        target_size=target_size,
        batch_size=batch_size,
        generator_params=generator_params
    ).load_generator(data_dir)


class GeneratorLoader(object):
    def __init__(self, target_size=(256, 256), batch_size=32, generator_params=None):
        self.batch_size = batch_size
        self.target_size = target_size
        if generator_params is None:
            self.generator_params = {}
        else:
            self.generator_params = generator_params

    def load_generator(self, data_dir):
        classes_ = get_classes(data_dir)
        class_mode = 'categorical' #binary return 1D binary labels

        data_gen = ImageDataGenerator(**self.generator_params)
        return data_gen.flow_from_directory(
            data_dir,
            target_size=self.target_size,
            classes=classes_,
            class_mode=class_mode,
            batch_size=self.batch_size)


def transform_values(dataframe, func):
    return pd.DataFrame(func(dataframe.values), columns=dataframe.columns, index=dataframe.index)


def split_images(data_dir, test_size=None, train_size=None):
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
