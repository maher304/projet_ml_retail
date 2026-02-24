import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.decomposition import PCA

# =========================================
# 1️⃣ Charger dataset nettoyé
# =========================================

DATA_PATH = "data/processed/retail_cleaned.csv"

df = pd.read_csv(DATA_PATH)

print("Shape dataset :", df.shape)

print("\nCorrélation avec Churn :")
print(df.corr(numeric_only=True)["Churn"].sort_values(ascending=False))

# =========================================
# 2️⃣ Corrélation & suppression features corrélées
# =========================================

numeric_df = df.select_dtypes(include=["int64", "float64"])

corr_matrix = numeric_df.corr()

# Heatmap (optionnel)
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Supprimer features avec corr > 0.9
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.9)]

print("Features supprimées (corr>0.9):", to_drop)

df = df.drop(columns=to_drop)

# =========================================
# 3️⃣ Séparer X / y
# =========================================

target = "Churn"

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================
# 4️⃣ Colonnes numériques / catégorielles
# =========================================

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns

# =========================================
# 5️⃣ Preprocessing + PCA
# =========================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95))  # garde 95% variance
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# =========================================
# 6️⃣ Modèle
# =========================================

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# =========================================
# 7️⃣ Training
# =========================================

pipeline.fit(X_train, y_train)

# =========================================
# 8️⃣ Évaluation
# =========================================

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:,1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("ROC-AUC :", roc_auc_score(y_test, y_prob))

# =========================================
# 9️⃣ Sauvegarde
# =========================================

joblib.dump(pipeline, "models/model_advanced.pkl")

print("\nModel saved successfully ✅")