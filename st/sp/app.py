import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf  # Make sure tensorflow is imported

start = '2010-01-01'
end = '2025-01-01'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter stock ticker', 'TSLA')
df = yf.download(user_input, start=start, end=end)

st.subheader('Data from 2010 - 2025')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, label='100-Day Moving Average')
plt.plot(ma200, label='200-Day Moving Average')
plt.plot(df['Close'], label='Closing Price')
plt.legend()
st.pyplot(fig)

# Splitting Data
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

# Fit scaler only on the training data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on training data
model.fit(x_train, y_train, batch_size=32, epochs=10)

# Preparing test data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

# Reshape x_test to add the 3rd dimension (features dimension, needed for LSTM)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], 100, 1))

# Predicting stock prices
y_predicted = model.predict(x_test)

# Reverse the scaling
scaler_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scaler_factor
y_test = np.array(y_test) * scaler_factor

# Plot Prediction vs Original
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
