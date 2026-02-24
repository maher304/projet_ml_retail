# src/utils.py

import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def save_model(model, path):
    import joblib
    joblib.dump(model, path)

def load_model(path):
    import joblib
    return joblib.load(path)