from sklearn.metrics import classification_report, roc_auc_score
import joblib
from src.config import MODEL_PATH

def evaluate(X_test, y_test):

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))