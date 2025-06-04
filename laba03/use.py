import keras
import cv2 as cv
import numpy as np

# === Завантаження збереженої моделі ===
model = keras.models.load_model("model2_lenet.keras")

# === Завантаження та обробка зображення ===
image_path = "image.png"
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
img_resized = cv.resize(img, (28, 28))
img_ready = img_resized.reshape((1, 28, 28, 1)) / 255.0

# === Передбачення ===
prediction = model.predict(img_ready)
predicted_digit = np.argmax(prediction)

print("Розпізнана цифра:", predicted_digit)