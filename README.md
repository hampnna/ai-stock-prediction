# 🤖 AI-Powered Stock Prediction System

A comprehensive machine learning system that predicts stock price movements using multiple data sources and advanced AI techniques.

## ✨ Features

- **📊 Multi-Source Data Fusion**: Historical prices + News sentiment + Social media sentiment
- **🧠 Advanced AI Models**: XGBoost classifier with FinBERT and VADER sentiment analysis
- **📈 Interactive Dashboard**: Real-time Streamlit web application with Plotly charts
- **🔍 Model Explainability**: SHAP values for feature importance analysis
- **💰 Backtesting Framework**: Realistic performance testing with transaction costs

## 🛠️ Tech Stack

- **Python 3.12**
- **Machine Learning**: XGBoost, scikit-learn
- **NLP**: FinBERT (Transformers), VADER Sentiment
- **Data**: yfinance, pandas, numpy
- **Visualization**: Streamlit, Plotly, matplotlib
- **Explainability**: SHAP

## 🚀 Quick Start

1. **Clone the repository**
git clone https://github.com/hampnna/ai-stock-prediction.git
cd ai-stock-prediction

2. **Create virtual environment**
python -m venv venv
venv\Scripts\activate # Windows

3. **Install dependencies**
pip install -r requirements.txt

4. **Run the data pipeline**
python src/data_ingest.py
python src/preprocess.py
python src/features.py
python src/model.py
python src/backtest.py

5. **Launch dashboard**
streamlit run src/app.py


## 📊 Project Structure

stock-prediction-project/
├── src/
│ ├── data_ingest.py # Data collection from Yahoo Finance
│ ├── preprocess.py # Technical indicators & sentiment analysis
│ ├── features.py # Feature engineering & target creation
│ ├── model.py # XGBoost training & validation
│ ├── backtest.py # Strategy backtesting
│ ├── explain.py # SHAP model explainability
│ └── app.py # Streamlit dashboard
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## 🎯 Key Results

- **Model Accuracy**: ~65-70% on next-day direction prediction
- **Features**: 67 engineered features including technical indicators and sentiment scores
- **Backtesting**: Includes realistic transaction costs (0.1%)
- **Explainability**: SHAP analysis reveals most important prediction factors

## 📈 Dashboard Features

- **Real-time stock price charts** with AI prediction markers
- **Technical analysis indicators** (RSI, MACD, Bollinger Bands)
- **Sentiment analysis trends** from news and social media
- **Portfolio performance comparison** (AI strategy vs buy-and-hold)
- **Feature importance rankings** with SHAP values

## 🔬 Model Details

- **Algorithm**: XGBoost Gradient Boosting Classifier
- **Features**: Price-based, technical indicators, sentiment scores
- **Validation**: Time series cross-validation (5 folds)
- **Target**: Binary classification (UP/DOWN next day movement)

## 🤝 Contributing

Feel free to open issues and pull requests!

## 📄 License

MIT License - feel free to use this code for learning and projects.

---

**Built with ❤️ by [hampanna]**

