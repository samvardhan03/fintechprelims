import pandas as pd
import yfinance as yf

def load_stock_data_yfinance(tickers, start_date='2018-01-01', end_date='2023-05-01'):
    stock_data = yf.download(tickers, start=start_date, end=end_date, group_by='Ticker', auto_adjust=True)
    return stock_data

def engineer_features(stock_data):
    # Assuming stock_data has a multi-index with levels ('Ticker', 'Date')
    # We need to get the 'Adj Close' column for each ticker
    adj_close = stock_data['Adj Close'].unstack('Ticker')

    # Create an empty DataFrame to store the engineered features
    engineered_data = pd.DataFrame()

    for ticker in adj_close.columns:
        ticker_data = adj_close[ticker]

        # Calculate returns
        ticker_data['returns'] = ticker_data.pct_change()

        # Calculate SMA_20 and SMA_50
        ticker_data['SMA_20'] = ticker_data.rolling(window=20).mean()
        ticker_data['SMA_50'] = ticker_data.rolling(window=50).mean()

        # Calculate RSI
        ticker_data['RSI'] = calculate_rsi(ticker_data)

        # Append the engineered features to the engineered_data DataFrame
        engineered_data = pd.concat([engineered_data, ticker_data], axis=1)

    return engineered_data
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
