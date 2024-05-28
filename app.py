pip install plotly
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from data_preprocessing import preprocess_text, extract_tickers
from models import load_model, make_predictions
from data_loader import load_stock_data_yfinance, engineer_features

def visualize_stock_data(stock_data, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=stock_data.index,
                                  open=stock_data['Open'],
                                  high=stock_data['High'],
                                  low=stock_data['Low'],
                                  close=stock_data['Close'],
                                  name=f'{ticker} Candlestick'))
    fig.update_layout(title=f'{ticker} Stock Price',
                      xaxis_title='Date',
                      yaxis_title='Price (USD)',
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

def main():
    st.title("Investment Recommendation Chatbot")
    user_input = st.text_input("Enter your investment query")

    if user_input:
        # Preprocess user input
        processed_input = preprocess_text(user_input)

        # Load data
        tickers = extract_tickers(user_input)
        if not tickers:
            st.write("No tickers found in the input. Please enter valid tickers.")
            return

        stock_data = load_stock_data_yfinance(tickers)
        if stock_data.empty:
            st.write(f"No stock data found for the tickers: {', '.join(tickers)}")
            return

        for ticker in tickers:
            visualize_stock_data(stock_data[ticker], ticker)

        stock_data = engineer_features(stock_data)

        # Load model
        model = load_model()

        # Make predictions
        X_new = stock_data[['returns', 'SMA_20', 'SMA_50', 'RSI']].values
        predictions = make_predictions(model, X_new)

        # Display recommendations
        st.write("Based on your query and the current market conditions, we recommend the following investments:")
        for ticker, prediction in zip(tickers, predictions):
            st.write(f"{ticker}: {prediction}")

if __name__ == "__main__":
    main()
