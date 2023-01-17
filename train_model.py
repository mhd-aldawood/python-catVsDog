import os

import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

path = r'C:\Users\Mohamad Aldawood\PycharmProjects\pythonProject\PetImages'

base_dir = 'dog-vs-cat-classification'

# Create datasets
train_datagen = tf.keras.preprocessing.image_dataset_from_directory(path,
                                                                    image_size=(200, 200),
                                                                    subset='training',
                                                                    seed=1,
                                                                    validation_split=0.1,
                                                                    batch_size=32)
test_datagen = tf.keras.preprocessing.image_dataset_from_directory(path,
                                                                   image_size=(200, 200),
                                                                   subset='validation',
                                                                   seed=1,
                                                                   validation_split=0.1,
                                                                   batch_size=32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', input_shape=(200, 200,3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

tf.data.experimental.ignore_errors()
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.fit_generator(train_datagen
                    , epochs=15,
                    validation_data=test_datagen
                    )

model.save("model.h5")
