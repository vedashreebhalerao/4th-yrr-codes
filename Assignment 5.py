# Aditya Kulkarni

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, RMSprop, Nadam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import time

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the neural network model
def create_model(optimizer):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Optimizers to be tested
optimizers = {
    'SGD': SGD(),
    'Adam': Adam(),
    'Adagrad': Adagrad(),
    'RMSprop': RMSprop(),
    'Nadam': Nadam()
}

# Store results
results = {}

# Train the model with different optimizers and record time, accuracy, and loss
for name, optimizer in optimizers.items():
    print(f"Training with {name} optimizer...")
    start_time = time.time()
    
    model = create_model(optimizer)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128, verbose=0)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    results[name] = {
        'history': history,
        'training_time': training_time
    }
    print(f"{name} optimizer took {training_time:.2f} seconds to converge.\n")

# Plot the results (Accuracy and Loss)
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Plot Accuracy
for name, result in results.items():
    axes[0].plot(result['history'].history['accuracy'], label=f"{name} Train Accuracy")
    axes[0].plot(result['history'].history['val_accuracy'], label=f"{name} Validation Accuracy")
axes[0].set_title('Accuracy Comparison')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

# Plot Loss
for name, result in results.items():
    axes[1].plot(result['history'].history['loss'], label=f"{name} Train Loss")
    axes[1].plot(result['history'].history['val_loss'], label=f"{name} Validation Loss")
axes[1].set_title('Loss Comparison')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.show()

# Display training time for each optimizer
print("Training Time for each Optimizer:")
for name, result in results.items():
    print(f"{name}: {result['training_time']:.2f} seconds")
