import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Load dataset
# -------------------------

df = pd.read_csv(
    "data/processed/exovision_dataset.csv"
)

print(df.head())

# -------------------------
# Features / Labels
# -------------------------

X = df.drop(
    columns=["target", "label"]
)

y = df["label"]

# -------------------------
# Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print()

print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))

# -------------------------
# Train
# -------------------------

model = RandomForestClassifier(

    n_estimators=300,

    random_state=42
)

model.fit(
    X_train,
    y_train
)

print()

print("Training Complete!")

# -------------------------
# Save
# -------------------------

joblib.dump(

    model,

    "models/random_forest.pkl"

)

print()

print("Model saved.")