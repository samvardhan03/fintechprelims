import streamlit as st
import pandas as pd
import numpy as np
from data_preprocessing import preprocess_text, extract_tickers
from models import load_model, make_predictions
from data_loader import load_stock_data_yfinance, engineer_features
import nltk
nltk.download('punkt')
nltk.download('all')


def main():
    st.title("Investment Recommendation Chatbot")
    user_input = st.text_input("Enter your investment query")

    if user_input:
        # Preprocess user input
        processed_input = preprocess_text(user_input)

        # Load data
        tickers = extract_tickers(user_input)
        stock_data = load_stock_data_yfinance(tickers)
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
