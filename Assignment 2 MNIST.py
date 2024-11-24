import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageEnhance, ImageOps, ImageTk

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the MNIST model
mnist_model = Sequential([
    Flatten(input_shape=(28, 28, 1)),  # Adding channel dimension
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

mnist_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mnist_model.fit(x_train[..., np.newaxis], y_train, validation_data=(x_test[..., np.newaxis], y_test), epochs=10, batch_size=128)

# GUI function to upload and predict on MNIST model
def upload_and_predict_mnist():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load image and ensure it's in grayscale
        img = Image.open(file_path).convert("L")  # Convert to grayscale

        # Enhance contrast to ensure the digit stands out
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)

        # Resize to 28x28 and invert colors if necessary
        img_resized = img.resize((28, 28))
        img_resized = ImageOps.invert(img_resized) if np.array(img_resized).mean() > 128 else img_resized

        # Display uploaded image on the screen
        img_display = ImageTk.PhotoImage(img_resized.resize((140, 140)))  # Resize for better display
        image_label.config(image=img_display)
        image_label.image = img_display

        # Normalize and reshape for model input
        img_array = np.array(img_resized).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)  # Add channel dimension for grayscale

        # Make prediction and display result
        prediction = np.argmax(mnist_model.predict(img_array), axis=1)
        result_label.config(text=f"Predicted Digit: {prediction[0]}")

# Tkinter GUI setup
root = tk.Tk()
root.title("Digit Recognition - MNIST")

# GUI elements
upload_button = tk.Button(root, text="Upload Digit Image", command=upload_and_predict_mnist)
upload_button.pack()

# Label for displaying uploaded image
image_label = tk.Label(root)
image_label.pack()

# Label for displaying prediction result
result_label = tk.Label(root, text="Prediction will appear here")
result_label.pack()

root.mainloop()
