import os

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


def load_generator(data_dir, target_size=(256, 256), batch_size=32, **generator_params):
    if generator_params is None:
        generator_params = {}
    classes_ = get_classes(data_dir)
    class_mode = 'binary' if len(classes_) <= 2 else 'categorical'

    data_gen = ImageDataGenerator(**generator_params)
    return data_gen.flow_from_directory(
        data_dir,
        target_size=target_size,
        classes=classes_,
        class_mode=class_mode,
        batch_size=batch_size)


def transform_values(dataframe, func):
    return pd.DataFrame(func(dataframe.values), columns=dataframe.columns, index=dataframe.index)
