import keras
import numpy as np
from keras.layers import Dense
import matplotlib.pyplot as plt

# === Параметри моделі ===
learning_rate = 0.1
nEpochs = 20
hidden_neurons = 2

# === Дані (логічна функція x or y or z) ===
features = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
])

labels = np.array([
    [0, 1],  # 0 or 0 or 0 = 0 (False)
    [1, 0],  # 1 or 0 or 0 = 1
    [1, 0],  # ...
    [1, 0],
    [1, 0],
    [1, 0],
    [1, 0],
    [1, 0]
])

# === Побудова моделі ===
initializer = keras.initializers.GlorotNormal(seed=12)
model = keras.Sequential([
    Dense(units=hidden_neurons, input_shape=(3,), kernel_initializer=initializer, activation='sigmoid'),
    Dense(units=2, kernel_initializer=initializer, activation='softmax')
])
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.95, nesterov=True)
)

# === Навчання ===
print('\n--- Навчання моделі ---')
history = model.fit(features, labels, epochs=nEpochs, batch_size=1, verbose=1)
print('Навчання завершено.\n')

# === Оцінювання ===
print('--- Оцінювання ---')
loss = model.evaluate(features, labels, verbose=0)
print(f"Loss: {loss:.4f}\n")

# === Прогнозування на всіх прикладах ===
print('--- Результати передбачення ---')
predictions = model.predict(features)
for i, pred in enumerate(predictions):
    input_vals = features[i]
    output = np.argmax(pred)
    logic_result = "True" if output == 0 else "False"
    print(f"{input_vals} => {logic_result} (вихід: {pred})")

# === Тестовий запит ===
print('\n--- Тест на [0, 0, 0] ---')
test = np.array([[0, 0, 0]])
test_pred = model.predict(test)
predicted_class = np.argmax(test_pred)
print(f"[0, 0, 0] => {'True' if predicted_class == 0 else 'False'} (вихід: {test_pred})")

# === Побудова графіку втрат ===
plt.plot(history.history['loss'])
plt.title("Зміна функції втрат (loss) під час навчання")
plt.xlabel("Епоха")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
