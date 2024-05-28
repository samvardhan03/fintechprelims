import pandas as pd
import yfinance as yf

def load_stock_data_yfinance(tickers, start_date='2018-01-01', end_date='2023-05-01'):
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    stock_data = stock_data.reset_index()
    stock_data['date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data.set_index('date')
    stock_data = stock_data.dropna()
    return stock_data['Adj Close']

def engineer_features(stock_data):
    stock_data['returns'] = stock_data.pct_change()
    stock_data['SMA_20'] = stock_data.rolling(window=20).mean()
    stock_data['SMA_50'] = stock_data.rolling(window=50).mean()
    stock_data['RSI'] = calculate_rsi(stock_data)
    return stock_data

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain, loss = delta.copy(), delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    gain = gain.ewm(com=window - 1, adjust=False).mean()
    loss = loss.abs().ewm(com=window - 1, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    return rsi


