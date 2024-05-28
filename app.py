import streamlit as st
import pandas as pd
import nltk
nltk.download('all')
import plotly.graph_objs as go
from data_preprocessing import preprocess_text, extract_tickers
from data_loader import load_stock_data_yfinance

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
    st.title("stock analysis bot")
    user_input = st.text_input("Enter the name of the stocks you wish to buy, the names of ticker codes of all stocks can be found here:'https://shorturl.at/TvQo8' ")

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
            stock_data = load_stock_data_yfinance(tickers)
            if stock_data.empty:
                st.write(f"No stock data found for the tickers: {', '.join(tickers)}")
                return

            for ticker in tickers:
                visualize_stock_data(stock_data[ticker], ticker)

if __name__ == "__main__":
    main()
