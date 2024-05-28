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

def extract_tickers_from_query(text):
    pattern = r"\b(?:[A-Z]+\s*[a-zA-Z]*\s*)+\b"
    tickers = re.findall(pattern, text.upper())
    return [ticker.replace(' ', '') for ticker in tickers if is_valid_ticker(ticker)]

def is_valid_ticker(ticker):
    # Add additional validation rules here
    # For now, we'll consider any uppercase string with optional spaces as a valid ticker
    return bool(re.match(r"^[A-Z]+(?:\s*[a-zA-Z]*)*$", ticker))
