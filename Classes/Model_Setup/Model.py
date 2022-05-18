from tensorflow import keras
from keras import layers

def model(classes):
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ])

    inputs = keras.Input(shape=(32,32,1))
    x = data_augmentation(inputs)

    x = layers.Conv2D(filters=16, kernel_size=3, padding='same', activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.05)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, padding='same', activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.05)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, padding='same', activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.05)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, padding='same', activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.05)(x)

    x = layers.Conv2D(filters=256, kernel_size=3, padding='same', activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.05)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model