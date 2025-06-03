import keras
import numpy as np
from keras.layers import Dense
import matplotlib.pyplot as plt

# --- Параметри ---
task = "XOR"  # або: "AND_Y_AND_Z", "OR_Y_OR_Z", "X_OR_Y_AND_Z", "X_AND_Y_AND_Z"
hidden_neurons = 4
learning_rate = 0.1
nEpochs = 500

# --- Вхідні дані ---
if task == "XOR":
    features = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    labels = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])

elif task == "X_AND_Y_AND_Z":
    features = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 1],
                         [1, 1, 0],
                         [0, 1, 1],
                         [0, 0, 1]])
    labels = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [1, 0]])

# --- Побудова моделі ---
initializer = keras.initializers.GlorotNormal(seed=42)
model = keras.Sequential([
    Dense(units=hidden_neurons, input_shape=(features.shape[1],), kernel_initializer=initializer, activation='sigmoid'),
    Dense(units=2, kernel_initializer=initializer, activation='softmax')
])
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True))

# --- Навчання ---
print(f"\n--- Навчання моделі для задачі: {task} ---")
history = model.fit(features, labels, epochs=nEpochs, batch_size=1, verbose=0)
print("Навчання завершено.")

# --- Оцінка ---
print("\nОцінка точності:")
loss = model.evaluate(features, labels, verbose=0)
print(f"Loss: {loss:.4f}")

# --- Прогнозування ---
print("\nПередбачення:")
predictions = model.predict(features)
for i, p in enumerate(predictions):
    logic_input = features[i]
    result = np.argmax(p)
    print(f"{logic_input} -> {result} ({'True' if result == 1 else 'False'})")

# --- Виведення конкретного випадку (XOR приклад) ---
if task == "XOR":
    print("\nРезультат для true XOR false:")
    res = model.predict(np.array([[1, 0]]))
    print(f"[1, 0] -> {np.argmax(res)} ({'True' if np.argmax(res) == 1 else 'False'})")

# --- Побудова графіку втрат ---
plt.plot(history.history['loss'])
plt.title("Графік втрат (Loss)")
plt.xlabel("Епохи")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
