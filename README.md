# Stock Price Prediction Using Machine Learning and Deep Learning: A CNN-LSTM Based Approach

This project uses CNN-LSTM hybrid models to predict stock prices. It includes feature engineering, technical indicators (SMA, RSI, MACD), and evaluation using RMSE and R² score.

## Features
- Real-time data with Yahoo Finance API
- Deep learning: CNN for feature extraction, LSTM for sequence prediction
- Comparison with traditional ML models like SVM and XGBoost


## Models Used
- Decision Tree (Scikit-learn)
- LSTM (Keras)
- CNN-LSTM (Keras)

## Folder Structure
```
stock-price-prediction-ml-dl/
├── dataset/
│   └── stock_data.csv
├── main.py
├── lstm_model.py
├── cnn_lstm_model.py
├── decision_tree_model.py
├── data_preprocessing.py
├── utils.py
├── requirements.txt
└── README.md
```
## Techniques Used
- Feature Engineering
- CNN for spatial pattern learning
- LSTM for temporal pattern detection
- Hybrid CNN-LSTM model for high-accuracy forecasting


## Results
The hybrid CNN-LSTM model performed better than traditional ML models, showcasing lower RMSE and better temporal accuracy on test data.

## Future Enhancements
- Add sentiment analysis from news
- Integrate technical indicators like RSI, MACD
- Deploy using Streamlit or Flask UI''',