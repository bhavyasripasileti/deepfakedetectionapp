from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

# Dummy LSTM model
model = Sequential()
model.add(LSTM(32, input_shape=(10, 10)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save the dummy model
model.save('deepfake_lstm_model.h5')
print("Dummy model saved as deepfake_lstm_model.h5")
