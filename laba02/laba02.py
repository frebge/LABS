mport keras
import numpy as np
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import csv

# ---------- Загальні функції ----------
def csv_reader(file_name):
    csv_dataset = pd.read_csv(file_name, names=["Label", "x", "y"])
    dataset = csv_dataset.copy()
    dataset_labels = keras.utils.to_categorical(dataset.pop('Label'))
    dataset_features = np.array(dataset)
    return dataset_features, dataset_labels

def predict_point(model, x, y):
    sample = np.array([[x, y]])
    pred = model.predict(sample)
    print(f"Point ({x}, {y}) => predicted class: {np.argmax(pred)}")

# ---------- Частина I: Бінарна класифікація ----------
print("=== PART I: BINARY CLASSIFICATION ===")
learning_rate = 0.1
nEpochs = 50
batch_size = 8

features_train, labels_train = csv_reader("saturn_data_train.csv")
features_val, labels_val = csv_reader("saturn_data_eval.csv")

keras.utils.set_random_seed(123)
initializer = keras.initializers.GlorotNormal(seed=12)

model_binary = keras.Sequential([
    Dense(input_shape=(2,), units=16, kernel_initializer=initializer, activation='relu'),
    Dense(units=2, kernel_initializer=initializer, activation='softmax')
])
model_binary.compile(loss='categorical_crossentropy',
                     optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.95, nesterov=True))

print('Training binary model...')
history_binary = model_binary.fit(features_train, labels_train, epochs=nEpochs, batch_size=batch_size, verbose=1)
print('Evaluation:')
model_binary.evaluate(features_val, labels_val)

plt.title("Binary classification loss")
plt.plot(history_binary.history['loss'])
plt.grid(True)
plt.show()

predict_point(model_binary, 0.5, 0.5)

# ---------- Частина II: Трикласова класифікація ----------
print("\n=== PART II: MULTICLASS CLASSIFICATION ===")

def create_multiclass_dataset(n_points=1000, filename="train_multiclass.csv"):
    data = []
    for _ in range(n_points):
        x, y = np.random.uniform(-5, 5), np.random.uniform(-5, 5)
        r1 = np.sqrt((x + 2)**2 + (y)**2)
        r2 = np.sqrt((x - 2)**2 + (y)**2)
        if r1 < 1.5:
            label = 0  # Клас 0: перше коло
        elif r2 < 1.5:
            label = 1  # Клас 1: друге коло
        else:
            label = 2  # Клас 2: решта площини
        data.append([label, x, y])
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

# Створити та зчитати дані
create_multiclass_dataset(1000, "train_multiclass.csv")
create_multiclass_dataset(300, "test_multiclass.csv")

features_train_m, labels_train_m = csv_reader("train_multiclass.csv")
features_test_m, labels_test_m = csv_reader("test_multiclass.csv")

model_multi = keras.Sequential([
    Dense(input_shape=(2,), units=32, kernel_initializer=initializer, activation='relu'),
    Dense(units=3, kernel_initializer=initializer, activation='softmax')
])
model_multi.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.Adam(learning_rate=0.01))

print("Training multiclass model...")
history_multi = model_multi.fit(features_train_m, labels_train_m, epochs=50, batch_size=16, verbose=1)

print("Evaluation:")
model_multi.evaluate(features_test_m, labels_test_m)

plt.title("Multiclass classification loss")
plt.plot(history_multi.history['loss'])
plt.grid(True)
plt.show()

predict_point(model_multi, 0.0, 0.0)
