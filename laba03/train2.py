from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import keras
import numpy as np

# === Завантаження та підготовка даних ===
(features_train, labels_train), (features_test, labels_test) = mnist.load_data()

features_train = features_train / 255
features_test = features_test / 255

labels_train_cat = keras.utils.to_categorical(labels_train, 10)
labels_test_cat = keras.utils.to_categorical(labels_test, 10)

# Розширення розмірності для 1 каналу (grayscale)
features_train = np.expand_dims(features_train, axis=3)
features_test = np.expand_dims(features_test, axis=3)

# === Побудова моделі LeNet з ядрами 3x3 та padding ===
model = Sequential([
    Conv2D(20, (3, 3), padding='same', strides=(1, 1), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(50, (3, 3), padding='same', strides=(1, 1), activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(500, activation='relu'),
    Dense(10, activation='softmax')
])

# === Компіляція та навчання ===
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(features_train, labels_train_cat, epochs=10, batch_size=64, validation_data=(features_test, labels_test_cat))

# === Оцінка та збереження ===
model.evaluate(features_test, labels_test_cat)
model.save("model2_lenet.keras")