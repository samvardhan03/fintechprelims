import pandas as pd
import yfinance as yf

def load_stock_data_yfinance(tickers, start_date='2018-01-01', end_date='2023-05-01'):
    stock_data = yf.download(tickers, start=start_date, end=end_date, group_by='Ticker', auto_adjust=True)
    return stock_data

def engineer_features(stock_data):
    engineered_data = pd.DataFrame()

    for column in stock_data.columns:
        if 'Adj Close' in column:
            ticker = column.split('_')[0]  # Extract the ticker from the column name
            ticker_data = stock_data[column]

            # Calculate returns
            ticker_data['returns'] = ticker_data.pct_change()

            # Calculate SMA_20 and SMA_50
            ticker_data['SMA_20'] = ticker_data.rolling(window=20).mean()
            ticker_data['SMA_50'] = ticker_data.rolling(window=50).mean()

            # Calculate RSI
            ticker_data['RSI'] = calculate_rsi(ticker_data)

            # Rename the columns to remove the 'Adj Close' part
            ticker_data.columns = [f"{ticker}_{col}" if col != 'Adj Close' else ticker for col in ticker_data.columns]

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
