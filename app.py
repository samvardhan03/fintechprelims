import streamlit as st
import pandas as pd
import nltk
nltk.download('all')
import plotly.graph_objs as go
from data_preprocessing import preprocess_text, extract_tickers
import yfinance as yf

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

def load_stock_data_yfinance(tickers, start_date='2018-01-01', end_date='2023-05-01'):
    stock_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    return stock_data

def main():
    st.title("Investment Recommendation Chatbot")
    user_input = st.text_input("Enter your investment query")

    if user_input:
        # Preprocess user input
        processed_input = preprocess_text(user_input)

        # Extract tickers from the input
        tickers = extract_tickers(user_input)

        if not tickers:
            # Handle general investment queries
            st.write("Based on your query, it seems like a general investment question. Here are some general investment tips:")
            st.write("- Diversify your portfolio across different asset classes and sectors.")
            st.write("- Consider your risk tolerance and investment horizon.")
            st.write("- Regularly review and rebalance your portfolio.")
            st.write("- Stay disciplined and invest regularly.")
        else:
            try:
                stock_data = load_stock_data_yfinance(tickers)
            except Exception as e:
                st.write(f"Error: {e}")
                return

            if not stock_data.empty:
                for ticker in tickers:
                    if ticker in stock_data.columns:
                        visualize_stock_data(stock_data[ticker], ticker)
                    else:
                        st.write(f"No stock data found for the ticker: {ticker}")
            else:
                st.write(f"No stock data found for the tickers: {', '.join(tickers)}")

if __name__ == "__main__":
    main()
