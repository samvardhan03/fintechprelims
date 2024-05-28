import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

def load_model():
    model_path = 'models/model.pkl'
    if os.path.isfile(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        # Create a new model instance if the file is not found
        model = RandomForestRegressor()
    return model

def make_predictions(model, X_new):
    predictions = model.predict(X_new)
    return predictions
