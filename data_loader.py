import pandas as pd
import yfinance as yf

def load_stock_data_yfinance(tickers, start_date='2018-01-01', end_date='2023-05-01'):
    stock_data = yf.download(tickers, start=start_date, end=date=end_date, group_by='Ticker', auto_adjust=True)
    return stock_data

def engineer_features(stock_data):
    stock_data = stock_data['Adj Close'].unstack(level=0)
    stock_data = stock_data.dropna()

    for ticker in stock_data.columns:
        stock_data[f'{ticker}_returns'] = stock_data[ticker].pct_change()
        stock_data[f'{ticker}_SMA_20'] = stock_data[ticker].rolling(window=20).mean()
        stock_data[f'{ticker}_SMA_50'] = stock_data[ticker].rolling(window=50).mean()
        stock_data[f'{ticker}_RSI'] = calculate_rsi(stock_data[ticker])

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
