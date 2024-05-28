
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def load_model():
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def make_predictions(model, X_new):
    predictions = model.predict(X_new)
    return predictions


