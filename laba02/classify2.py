import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential

def read_csv_data(path):
    df = pd.read_csv(path, names=['Target', 'feat1', 'feat2'])
    y_data = keras.utils.to_categorical(df.pop('Target'))
    x_data = np.array(df)
    return x_data, y_data

def build_model(input_dim, hidden_units, output_classes, lr=0.005):
    model = Sequential()
    init = keras.initializers.GlorotNormal(seed=12)
    model.add(Dense(hidden_units, input_shape=(input_dim,), activation='relu', kernel_initializer=init))
    model.add(Dense(output_classes, activation='softmax', kernel_initializer=init))
    optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.95, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    return model

if __name__ == '__main__':
    # Гіперпараметри
    lr = 0.005
    n_epochs = 200
    batch = 32  # Уменьшенный размер batch для лучшей сходимости
    hidden_layer_neurons = 16

    # Зчитування датасетів
    X_train, Y_train = read_csv_data('saturn_data_train.csv')
    X_test, Y_test = read_csv_data('saturn_data_eval.csv')  # Убран пробел в имени файла

    keras.utils.set_random_seed(42)

    # Створення та навчання моделі
    classifier = build_model(input_dim=2, hidden_units=hidden_layer_neurons, output_classes=3, lr=lr)
    print(" Навчання розпочато...")
    history = classifier.fit(X_train, Y_train, epochs=n_epochs, batch_size=batch, verbose=1)
    print(" Навчання завершено!")

    # Оцінювання
    print("\n📊 Результат оцінювання:")
    classifier.evaluate(X_test, Y_test)

    # Збереження моделі
    classifier.save("custom_saturn_model.keras")

    # Тестове передбачення
    test_point = np.array([[2, 3]])
    prediction = classifier.predict(test_point)
    print("\n🔍 Прогноз для точки (2, 3):", prediction)
    print("Належить до класу:", np.argmax(prediction))

    # Візуалізація loss-графіку
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], color='blue')
    plt.title("Графік втрат під час навчання")
    plt.xlabel("Епоха")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # Візуалізація навчального датасету
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1],
                         c=np.argmax(Y_train, axis=1),
                         cmap='plasma',
                         alpha=0.6)
    plt.colorbar(scatter)
    plt.title("Навчальний датасет (розфарбовано по класах)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()
