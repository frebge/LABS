from keras.models import load_model
import numpy as np

def classify_point(x_val, y_val):
    model = load_model("custom_saturn_model.keras")
    test = np.array([[x_val, y_val]])
    prediction = model.predict(test)
    return np.argmax(prediction)

if __name__ == '__main__':
    print("Point (10, 2) -> Class:", classify_point(10, 2))
    print("Point (2, 0) -> Class:", classify_point(2, 0))
    print("Point (17, 0) -> Class:", classify_point(17, 0))