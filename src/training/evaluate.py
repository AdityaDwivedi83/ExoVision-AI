import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from sklearn.model_selection import train_test_split

# -------------------------
# Load dataset
# -------------------------

df = pd.read_csv(
    "data/processed/exovision_dataset.csv"
)

X = df.drop(
    columns=["target", "label"]
)

y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# Load trained model
# -------------------------

model = joblib.load(
    "models/random_forest.pkl"
)

pred = model.predict(X_test)

prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy :", accuracy_score(y_test, pred))

print("Precision:", precision_score(y_test, pred))

print("Recall   :", recall_score(y_test, pred))

print("F1 Score :", f1_score(y_test, pred))

print("ROC AUC  :", roc_auc_score(y_test, prob))

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, pred))

print("\nClassification Report\n")

print(classification_report(y_test, pred))