import keras
import numpy as np
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd


# створення набору даних, зчитаних з csv файлу
def csv_reader(file_name):
    csv_dataset = pd.read_csv(
        file_name,
        names=["Label", "x", "y"])
    # csv_dataset.head()
    dataset = csv_dataset.copy()
    dataset_labels = keras.utils.to_categorical(
        dataset.pop('Label'))  # перетворює у one-hot-encoding подання, у бінарний вектор для кожного значення
    dataset_features = np.array(dataset)
    return dataset_features, dataset_labels


learning_rate = ???
nEpochs = 10
batch_size = ???

# Завантаження даних для навчання і оцінювання
features_train, labels_train = csv_reader(
    "..\\saturn_data_train.csv")
features_val, labels_val = csv_reader(
    "..\\saturn_data_eval.csv")

keras.utils.set_random_seed(123) #для відтворюваності результатів
initializer = keras.initializers.GlorotNormal(seed=12)  # =Xavier
model = keras.Sequential([
    Dense(input_shape=???, units=???, kernel_initializer=initializer, activation='relu'),
    Dense(units=???, kernel_initializer=initializer, activation='softmax')
])
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.95, nesterov=True))
print('learning start')
history = model.fit(features_train, labels_train, epochs=nEpochs, batch_size=batch_size,
                    verbose=1)  # , use_multiprocessing = True, workers = 16)
print('learning finish')
print('evaluation:')
model.evaluate(features_val, labels_val)
print('predict=')
# За допомогою моделі визначити, якому класу належить точка 

plt.plot(history.history['loss'])
plt.grid(True)
plt.show()
