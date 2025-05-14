import numpy as np
from utils import load_data, scale_data
from data_preprocessing import create_dataset
from lstm_model import build_lstm_model
from cnn_lstm_model import build_cnn_lstm_model
from decision_tree_model import train_decision_tree
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = load_data("dataset/stock_data.csv")
data = df[['Close']].values
scaled_data, scaler = scale_data(data)
X, y = create_dataset(scaled_data, time_step=60)
X = X.reshape(X.shape[0], X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train LSTM
lstm_model = build_lstm_model(X_train.shape)
lstm_model.fit(X_train, y_train, epochs=5, batch_size=32)
print("LSTM Test Loss:", lstm_model.evaluate(X_test, y_test))

# Train CNN-LSTM
cnn_model = build_cnn_lstm_model(X_train.shape)
cnn_model.fit(X_train, y_train, epochs=5, batch_size=32)
print("CNN-LSTM Test Loss:", cnn_model.evaluate(X_test, y_test))

# Train Decision Tree
X_dt = X.reshape(X.shape[0], X.shape[1])  # Flatten for tree
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_dt, y, test_size=0.2, shuffle=False)
dt_model = train_decision_tree(X_train_dt, y_train_dt)
print("Decision Tree Score:", dt_model.score(X_test_dt, y_test_dt))
