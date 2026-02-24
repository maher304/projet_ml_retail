import joblib
import pandas as pd
from src.config import MODEL_PATH

model = joblib.load(MODEL_PATH)

def predict(data_dict):
    df = pd.DataFrame([data_dict])
    return model.predict(df)