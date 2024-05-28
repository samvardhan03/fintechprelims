
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
from data_loader import load_stock_data_yfinance, engineer_features

# Load data
tickers = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
stock_data = load_stock_data_yfinance(tickers)
stock_data = engineer_features(stock_data)

# Split data into features and target
X = stock_data[['returns', 'SMA_20', 'SMA_50', 'RSI']].values
y = stock_data['Adj Close'].shift(-1).values

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
