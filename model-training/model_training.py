import data_loading
import model_definition
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(X_train, y_train), (X_validation, y_validation) = data_loading.load_train_data()

if K.image_data_format == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_validation = X_validation.reshape(X_validation.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_validation = X_validation.reshape(X_validation.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

train_data_augmentation = ImageDataGenerator(
    rescale=1 / 255.,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

validation_data_augmentation = ImageDataGenerator(rescale=1 / 255.)

train_data_generator = train_data_augmentation.flow(X_train, y_train)
validation_data_generator = validation_data_augmentation.flow(X_validation, y_validation)

model = model_definition.get_model(input_shape, 'adam', 'categorical_crossentropy', ['accuracy'])

history = model.fit(train_data_generator,
                    steps_per_epoch=int(len(X_train) / 32),
                    epochs=10,
                    validation_data=validation_data_generator,
                    validation_steps=int(len(X_validation) / 32))

