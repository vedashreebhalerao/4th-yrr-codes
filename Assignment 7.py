# Aditya Kulkarni

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, Dropout

# Load dataset
df = pd.read_csv("Assets/BTC_USD.csv", parse_dates=['date'])
df = df.sort_values('date')
df.reset_index(drop=True, inplace=True)

# Data Preprocessing
# Using MinMaxScaler to scale data between 0 and 1
scaler = MinMaxScaler()
close_price = df[['close']].values
scaled_close = scaler.fit_transform(close_price)

# Define sequence length and prepare data sequences for RNN
SEQ_LEN = 60  # sequence length (60 days)
def to_sequences(data, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

X, y = to_sequences(scaled_close)

# Split data into training and testing sets (80% train, 20% test)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Model Architecture
model = Sequential()

# Adding Bidirectional LSTM layer
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

# Adding GRU layer
model.add(GRU(32, return_sequences=False))
model.add(Dropout(0.2))

# Dense layer for output
model.add(Dense(1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Model evaluation
model.evaluate(X_test, y_test)

# Plotting training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Making Predictions
predictions = model.predict(X_test)
predicted_price = scaler.inverse_transform(predictions)
actual_price = scaler.inverse_transform(y_test)

# Plotting Actual vs Predicted Prices
plt.plot(actual_price, color='blue', label='Actual Price')
plt.plot(predicted_price, color='red', label='Predicted Price')
plt.title('Cryptocurrency Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
