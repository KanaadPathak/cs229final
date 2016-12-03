from cnn_feature import ClassifierPool, CNNFeatureExtractor

train_feature = 'history/imageclef_train_resnet50.h5'
test_feature = 'history/imageclef_test_resnet50.h5'

X_train, y_train, train_classes = CNNFeatureExtractor().load_features(feature_file=train_feature)
reverse = dict(zip(train_classes.values(), train_classes.keys()))
X_test, y_test, test_classes = CNNFeatureExtractor().load_features(feature_file=test_feature)
y_test = [reverse[test_classes[label]] for label in y_test]



