from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import csv
import zipfile
import os
import shutil

from xml.dom import minidom

from preprocess_utils import transform_values


class KaggleLoader(object):
    def __init__(self, csv_file, label_column='species', scale=True):
        # Load Data
        train = pd.read_csv(csv_file)

        # Label Encoding
        self.encoder = LabelEncoder()
        if label_column is not None:
            self.y = self.encoder.fit_transform(train[label_column])
            train = train.drop([label_column], axis=1)

        # Reshape the data
        self.X = train.set_index('id', drop=True)
        self.X.index.name = None

        # Standard Scaler (regularization)
        self.scaler = StandardScaler()
        if scale:
            self.X_train = transform_values(self.X, self.scaler.fit_transform)


class SwedishLoader(object):
    def __init__(self):
        self.name_dict = {
            1: "Ulmus carpinifolia",
            2: "Acer",
            3: "Salix aurita",
            4: "Quercus",
            5: "Alnus incana",
            6: "Betula pubescens",
            7: "Salix alba 'Sericea'",
            8: "Populus tremula",
            9: "Ulmus glabra",
            10:  "Sorbus aucuparia",
            11:  "Salix sinerea",
            12:  "Populus",
            13:  "Tilia",
            14:  "Sorbus intermedia",
            15:  "Fagus silvatica"
        }

    def extract(self):
        f = open('meta.csv', 'w')
        f.write('name, scientific_name\n')
        for k, v in self.name_dict.items():
            f.write('%s, %s\n' % (k, v))

        for i in self.name_dict.keys():
            filename = "leaf%i.zip" % i
            print("extracting", filename)
            z = zipfile.ZipFile(filename, 'r')
            z.extractall(str(i))
            z.close()

    def split(self, origin_path, train_path, val_path):
        for i in self.name_dict.keys():
            for j in range(1, 75):
                filename = 'l%inr%3i.tif' % (i, j)
                if j <= 55:
                    path = '%s/%s' % (train_path, j)
                else:
                    path = '%s/%s' % (val_path, j)
                os.makedirs(path, exist_ok=True)
                os.rename('%s/%i/%s' % (origin_path, j, filename),
                          '%s/%s' % (path, filename))


class FlaviaLoader(object):
    def __init__(self):
        cols = ['scientific_name', 'common_names', 'filename', 'url']
        df = pd.read_csv('meta.csv', sep='|', skipinitialspace=True)
        df.columns = [x.strip().replace(' ', '_') for x in df.columns]
        df[cols] = df[cols].astype(str).apply(lambda x: x.str.rstrip())
        df = df.set_index('label', drop=True)
        df.index.name = None

    def load_labels(self):
        header = ['label', 'scientific_name', 'common_names', 'filename', 'url']
        with open('data/flavia/label.csv', 'w') as f1:
            with open('data/flavia/meta.md', 'r') as f2:
                reader = csv.reader(f2, skipinitialspace=True, delimiter='|')
                next(reader, None)  # skip the headers
                for row in reader:
                    label = row[header.index('label')].rstrip()
                    l, r = row[header.index('filename')].rstrip().split('-')
                    for j in range(int(l), int(r)+1):
                        f1.write('%s, %i\n' % (label, j))


class UCILoader(object):
    def __init__(self, csv_file):
        with open(csv_file, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                print(', '.join(row))

    def load_labels(self):
        pass


class ImageClefLoader(object):
    def extract_leaves(self, root_dir):
        # root_dir = sys.argv[1]
        image_dir = '%s/train' % root_dir
        for f in os.listdir(image_dir):
            if f.endswith('.xml'):
                filename = '%s/%s' % (image_dir, f)
                xmldoc = minidom.parse(filename)
                the_content = xmldoc.getElementsByTagName('Content')[0].firstChild.data
                the_type = xmldoc.getElementsByTagName('Type')[0].firstChild.data
                the_classid = xmldoc.getElementsByTagName('ClassId')[0].firstChild.data
                the_filename = xmldoc.getElementsByTagName('FileName')[0].firstChild.data
                if the_content == 'Leaf':
                    print(','.join([the_type, the_classid, the_content, the_filename]))
                    dest_dir = '%s/ready' % root_dir
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy('%s/%s' % (image_dir, the_filename), '%s/%s' % (dest_dir, the_filename))
