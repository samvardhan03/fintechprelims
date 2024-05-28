import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

def extract_tickers(text):
    pattern = r"\b[A-Z]+\b"
    tickers = re.findall(pattern, text.upper())
    return [ticker for ticker in tickers if is_valid_ticker(ticker)]

def is_valid_ticker(ticker):
    # Add additional validation rules here
    # For now, we'll consider any uppercase string as a valid ticker
    return True
