from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense
import keras
import numpy as np

# === Завантаження та підготовка даних ===
(features_train, labels_train), (features_test, labels_test) = mnist.load_data()

features_train = features_train / 255
features_test = features_test / 255

labels_train_cat = keras.utils.to_categorical(labels_train, 10)
labels_test_cat = keras.utils.to_categorical(labels_test, 10)

# === Побудова моделі ===
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(500, activation='relu'),
    Dense(10, activation='softmax')
])

# === Компіляція та навчання ===
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(features_train, labels_train_cat, epochs=15, batch_size=64, validation_data=(features_test, labels_test_cat))

# === Оцінка та збереження ===
model.evaluate(features_test, labels_test_cat)
model.save("model1_mlp.keras")