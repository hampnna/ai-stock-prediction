from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Reshape

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Reshape((input_shape[1], 1), input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model