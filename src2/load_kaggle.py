import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import argparse


class KaggleLoader(object):
    def __init__(self, csv_file, label_column=None, scale=True):
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


def transform_values(df, func):
    return pd.DataFrame(func(df.values), columns=df.columns, index=df.index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_file', help="csv file from kaggle")
    args = parser.parse_args()
    data_set = KaggleLoader(args.csv_file, 'species')
    print(data_set.X_train)
