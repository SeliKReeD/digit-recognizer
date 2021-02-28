from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Flatten, MaxPooling2D
from tensorflow.keras import backend as K


def get_model(input_shape, optimizer, loss_function, metrics):
    model = Sequential([
        Conv2D(64, kernel_size=(7, 7), activation="relu", padding="same", input_shape=input_shape),
        MaxPooling2D(2),
        Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
        Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
        MaxPooling2D(2),
        Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
        Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
        MaxPooling2D(2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax")
        # Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        # MaxPooling2D((2, 2)),
        # Dropout(0.25),
        #
        # Conv2D(64, kernel_size=(3, 3), activation='relu'),
        # MaxPooling2D((2, 2)),
        # Dropout(0.25),
        #
        # Flatten(),
        #
        # Dense(1024, activation='relu'),
        # Dropout(0.25),
        #
        # Dense(10, activation='softmax')
    ])

    model.compile(optimizer, loss_function, metrics=metrics)

    return model


if K.image_data_format() == 'channels_first':
    i_shape = (1, 28, 28)
else:
    i_shape = (28, 28, 1)
model_temp = get_model(i_shape, 'adam', 'categorical_crossentropy', ['accuracy'])

model_temp.summary()
