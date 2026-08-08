import numpy as np
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.deep_learning.dataset import ExoVisionDataset
from src.deep_learning.cnn_model import ExoVisionCNN


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/exovision_cnn.pth"

BATCH_SIZE = 8


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = ExoVisionDataset()


# IMPORTANT:
# Use the same validation split as training.

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


indices = list(range(len(dataset)))

labels = dataset.data["label"].values

_, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

val_dataset = Subset(
    dataset,
    val_indices
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(
    "Validation samples:",
    len(val_dataset)
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = ExoVisionCNN()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# --------------------------------------------------
# Predictions
# --------------------------------------------------

all_labels = []

all_probabilities = []

all_predictions = []


with torch.no_grad():

    for X, y in val_loader:

        X = X.to(device)

        outputs = model(X)

        probabilities = torch.sigmoid(
            outputs
        )

        predictions = (
            probabilities >= 0.5
        ).int()

        all_labels.extend(
            y.numpy()
        )

        all_probabilities.extend(
            probabilities.cpu()
            .numpy()
            .flatten()
        )

        all_predictions.extend(
            predictions.cpu()
            .numpy()
            .flatten()
        )


# Convert to arrays

y_true = np.array(
    all_labels
).astype(int)

y_prob = np.array(
    all_probabilities
)

y_pred = np.array(
    all_predictions
).astype(int)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_true,
    y_prob
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n==============================")
print("ExoVision CNN Evaluation")
print("==============================")

print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC AUC   : {auc:.4f}"
)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix")

print(cm)


# --------------------------------------------------
# Classification Report
# --------------------------------------------------

print("\nClassification Report")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)