import pandas as pd
import numpy as np

INPUT_PATH = "data/raw/retail_customers_COMPLETE_CATEGORICAL.csv"
OUTPUT_PATH = "data/processed/retail_cleaned.csv"

def clean_dataset():

    df = pd.read_csv(INPUT_PATH)

    print("Shape initiale :", df.shape)

    # =============================
    # 1️⃣ Supprimer colonnes inutiles
    # =============================

    cols_to_drop = ["Newsletter"]  # variance nulle
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # =============================
    # 2️⃣ Parsing date inscription
    # =============================

    if "RegistrationDate" in df.columns:
        df["RegistrationDate"] = pd.to_datetime(df["RegistrationDate"], errors="coerce")
        df["RegYear"] = df["RegistrationDate"].dt.year
        df["RegMonth"] = df["RegistrationDate"].dt.month
        df = df.drop(columns=["RegistrationDate"])

    # =============================
    # 3️⃣ Feature Engineering
    # =============================

    if "MonetaryTotal" in df.columns and "Recency" in df.columns:
        df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)

    if "MonetaryTotal" in df.columns and "Frequency" in df.columns:
        df["AvgBasketValue"] = df["MonetaryTotal"] / (df["Frequency"] + 1)

    if "Recency" in df.columns and "CustomerTenureDays" in df.columns:
        df["TenureRatio"] = df["Recency"] / (df["CustomerTenureDays"] + 1)

    # =============================
    # 4️⃣ Gestion valeurs aberrantes
    # =============================

    # SupportTickets
    if "SupportTickets" in df.columns:
        df.loc[df["SupportTickets"] < 0, "SupportTickets"] = np.nan
        df.loc[df["SupportTickets"] > 100, "SupportTickets"] = np.nan

    # Satisfaction
    if "Satisfaction" in df.columns:
        df.loc[df["Satisfaction"] < 0, "Satisfaction"] = np.nan
        df.loc[df["Satisfaction"] > 5, "Satisfaction"] = np.nan

    # Quantités négatives extrêmes
    if "TotalQuantity" in df.columns:
        df.loc[df["TotalQuantity"] < 0, "TotalQuantity"] = np.nan

    # =============================
    # 5️⃣ Gestion valeurs manquantes simples
    # =============================

    # Numériques → médiane
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Catégorielles → mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("Shape après nettoyage :", df.shape)

    # =============================
    # 6️⃣ Sauvegarde
    # =============================

    df.to_csv(OUTPUT_PATH, index=False)
    print("Dataset nettoyé sauvegardé ✅")

if __name__ == "__main__":
    clean_dataset()