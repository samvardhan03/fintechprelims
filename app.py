import streamlit as st
import pandas as pd
from data_preprocessing import preprocess_text, extract_tickers
from models import load_model, make_predictions
from data_loader import load_stock_data_yfinance, engineer_features

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
