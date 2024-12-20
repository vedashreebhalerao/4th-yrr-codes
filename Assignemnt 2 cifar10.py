from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical  # Add this import line
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image , ImageTk

# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the CIFAR-10 model
cifar_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

cifar_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cifar_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=64)

cifar_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# GUI for uploading an image and testing the CIFAR-10 model
def upload_and_predict_cifar():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load image and resize to 32x32
        img = Image.open(file_path).convert("RGB")  # CIFAR-10 expects RGB images
        img_resized = img.resize((32, 32))

        # Display uploaded image on the screen
        img_display = ImageTk.PhotoImage(img_resized.resize((140, 140)))  # Resize for better display in GUI
        image_label.config(image=img_display)
        image_label.image = img_display

        # Normalize and reshape for model input
        img_array = np.array(img_resized).astype("float32") / 255.0
        img_array = img_array.reshape(1, 32, 32, 3)  # CIFAR-10 input shape

        # Make prediction and display result
        prediction = np.argmax(cifar_model.predict(img_array), axis=1)
        result_label.config(text=f"Predicted Class: {cifar_labels[prediction[0]]}")

# Tkinter GUI setup
root = tk.Tk()
root.title("Image Recognition - CIFAR-10")

# GUI elements
upload_button = tk.Button(root, text="Upload Image", command=upload_and_predict_cifar)
upload_button.pack()

# Label for displaying uploaded image
image_label = tk.Label(root)
image_label.pack()

# Label for displaying prediction result
result_label = tk.Label(root, text="Prediction will appear here")
result_label.pack()

root.mainloop()
