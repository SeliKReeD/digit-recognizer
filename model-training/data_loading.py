import csv
import numpy as np
import configuration
from tensorflow.keras.utils import to_categorical


def load_train_data():
    features = []
    labels = []
    with open(configuration.train_data_file_path) as train_csv_file:
        train_csv_reader = csv.reader(train_csv_file, delimiter=',')
        next(train_csv_reader)
        for row in train_csv_reader:
            features.append(row[1:])
            labels.append(row[0])

    labels = to_categorical(labels, num_classes=10)

    features = np.array(features).astype(float)
    labels = np.array(labels).astype(float)

    split_index = int(len(features) * configuration.validation_split_ratio)

    validation_features = features[split_index:]
    validation_labels = labels[split_index:]

    train_features = features[:split_index]
    train_labels = labels[:split_index]

    print("Loaded features array from train file with shape: ", features.shape)
    print("Loaded labels array from train file with shape: ", labels.shape)

    print("Train features array with shape: ", train_features.shape)
    print("Train labels array with shape: ", train_labels.shape)
    print("Validation features array with shape: ", validation_features.shape)
    print("Validation labels array with shape: ", validation_labels.shape)

    return (train_features, train_labels), (validation_features, validation_labels)


def load_test_data():
    features = []
    with open(configuration.test_data_file_path) as test_csv_file:
        test_csv_reader = csv.reader(test_csv_file, delimiter=',')
        next(test_csv_reader)
        for row in test_csv_reader:
            features.append(row)
    features = np.array(features).astype(float)
    return features / 255.
