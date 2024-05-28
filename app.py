import streamlit as st
import pandas as pd
import nltk
nltk.download('all')
import plotly.graph_objs as go
from data_preprocessing import preprocess_text, extract_tickers_from_query, is_stock_query
from models import load_model, make_predictions
from data_loader import load_stock_data_yfinance, engineer_features

def visualize_stock_data(stock_data, ticker):
    # ... (same as before)

def main():
    st.title("Investment Recommendation Chatbot")
    user_input = st.text_input("Enter your investment query")

    if user_input:
        # Preprocess user input
        processed_input = preprocess_text(user_input)

        # Check if the query is related to stocks
        if is_stock_query(processed_input):
            # Extract tickers from the query
            tickers = extract_tickers_from_query(user_input)

            if tickers:
                stock_data = load_stock_data_yfinance(tickers)
                if not stock_data.empty:
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
                else:
                    st.write(f"No stock data found for the tickers: {', '.join(tickers)}")
            else:
                st.write("Your query doesn't seem to contain any valid stock tickers. Please try again with a different query.")
        else:
            # Handle general investment queries
            st.write("Based on your query, it seems like a general investment question. Here are some general investment tips:")
            st.write("- Diversify your portfolio across different asset classes and sectors.")
            st.write("- Consider your risk tolerance and investment horizon.")
            st.write("- Regularly review and rebalance your portfolio.")
            st.write("- Stay disciplined and invest regularly.")

if __name__ == "__main__":
    main()
