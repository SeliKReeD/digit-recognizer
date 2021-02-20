import csv
import numpy as np
import configuration


def load_train_data(train_file_path, test_file_path):
    features = []
    labels = []
    validation = []
    with open(train_file_path) as train_csv_file:
        train_csv_reader = csv.reader(train_csv_file, delimiter=',')
        next(train_csv_reader)
        for row in train_csv_reader:
            features.append(row[1:])
            labels.append(row[0])

    features = np.array(features).astype(float)
    labels = np.array(labels).astype(float)

    with open(test_file_path) as test_csv_file:
        test_csv_reader = csv.reader(test_csv_file, delimiter=',')
        next(test_csv_reader)
        for row in test_csv_reader:
            validation.append(row)

    validation = np.array(validation).astype(float)

    print("Loaded features array from train file with shape: ", features.shape)
    print("Loaded labels array from train file with shape: ", labels.shape)

    print("Loaded validation array from test file with shape: ", validation.shape)
    return (features, labels), validation
