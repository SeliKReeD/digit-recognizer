from keras.models import load_model
from data_loading import load_test_data
from tensorflow.keras import backend as K
import configuration
import csv
import numpy as np


data = load_test_data()

if K.image_data_format == 'channels_first':
    X_train = data.reshape(data.shape[0], 1, 28, 28)
else:
    data = data.reshape(data.shape[0], 28, 28, 1)

model = load_model(f"{configuration.root_data_directory}/snapshots/28-02-2021-18-59-10-SNAPSHOT-l-0.027-a-0.994.h5")

predictions = model.predict(data)

with open('submition.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file, delimiter=',')
    counter = 1
    csv_writer.writerow(['ImageId', 'Label'])
    for prediction in predictions:
        csv_writer.writerow([counter, np.argmax(prediction)])
        counter += 1
