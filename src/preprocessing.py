import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.config import TEST_SIZE, RANDOM_STATE

def load_data(path):
    return pd.read_csv(path)

def feature_engineering(df):
    df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)
    df["AvgBasketValue"] = df["MonetaryTotal"] / (df["Frequency"] + 1)
    return df

def split_data(df, target="Churn"):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

def build_preprocessor(numeric_features, categorical_features):

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )