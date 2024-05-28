import streamlit as st
import pandas as pd
import nltk
import pickle
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objs as go
from data_preprocessing import preprocess_text, extract_tickers
from data_loader import load_stock_data_yfinance, engineer_features

nltk.download('all')

def load_model():
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def make_predictions_for_ticker(model, ticker_data, lookahead=5):
    features = ticker_data[['returns', 'SMA_20', 'SMA_50', 'RSI']].values[-lookahead:]
    predictions = model.predict(features)
    prediction_dates = pd.date_range(start=ticker_data.index[-1], periods=lookahead+1, closed='right')
    predictions_df = pd.Series(predictions, index=prediction_dates)
    return predictions_df

def visualize_stock_data(stock_data, ticker, predictions=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=stock_data.index,
                                 open=stock_data['Open'],
                                 high=stock_data['High'],
                                 low=stock_data['Low'],
                                 close=stock_data['Close'],
                                 name=f'{ticker} Candlestick'))
    
    if predictions is not None:
        fig.add_trace(go.Scatter(x=predictions.index,
                                 y=predictions,
                                 mode='lines',
                                 name='Predicted Close Price',
                                 line=dict(color='blue')))
    
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

            # Load your trained model
            model = load_model()

            for ticker in tickers:
                ticker_data = stock_data[ticker]
                engineered_data = engineer_features(ticker_data)

                # Make predictions
                predictions = make_predictions_for_ticker(model, engineered_data)

                # Visualize stock data and predictions
                visualize_stock_data(ticker_data, ticker, predictions)

if __name__ == "__main__":
    main()
