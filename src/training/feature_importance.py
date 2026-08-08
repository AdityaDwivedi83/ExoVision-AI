import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("data/processed/exovision_dataset.csv")

X = df.drop(columns=["target", "label"])

# ----------------------------
# Load Model
# ----------------------------

model = joblib.load("models/random_forest.pkl")

# ----------------------------
# Feature Importance
# ----------------------------

importance = model.feature_importances_

feature_importance = (
    pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })
    .sort_values(
        by="Importance",
        ascending=True
    )
)

print(feature_importance)

# ----------------------------
# Plot
# ----------------------------

plt.figure(figsize=(10,6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig("results/feature_importance.png")

plt.show()