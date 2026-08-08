import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.deep_learning.dataset import ExoVisionDataset
from src.deep_learning.cnn_model import ExoVisionCNN


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001

MODEL_PATH = "models/exovision_cnn.pth"


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

# --------------------------------------------------
# Stratified train / validation split
# --------------------------------------------------

indices = list(range(len(dataset)))

labels = dataset.data["label"].values

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

train_dataset = Subset(
    dataset,
    train_indices
)

val_dataset = Subset(
    dataset,
    val_indices
)

print()
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = ExoVisionCNN()

model = model.to(device)


# --------------------------------------------------
# Loss + Optimizer
# --------------------------------------------------

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training
# --------------------------------------------------

best_val_loss = float("inf")

os.makedirs(
    "models",
    exist_ok=True
)

print("\nStarting training...\n")


for epoch in range(EPOCHS):

    # ==============================================
    # Training
    # ==============================================

    model.train()

    train_loss = 0.0

    for X, y in train_loader:

        X = X.to(device)

        y = y.to(device)

        # Shape:
        # y = [batch]
        #
        # Model output:
        # [batch, 1]
        #
        # Make them compatible.

        y = y.unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(
            outputs,
            y
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item() * X.size(0)
        )

    train_loss /= len(
        train_loader.dataset
    )

    # ==============================================
    # Validation
    # ==============================================

    model.eval()

    val_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for X, y in val_loader:

            X = X.to(device)

            y = y.to(device)

            y = y.unsqueeze(1)

            outputs = model(X)

            loss = criterion(
                outputs,
                y
            )

            val_loss += (
                loss.item() * X.size(0)
            )

            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities >= 0.5
            ).float()

            correct += (
                predictions == y
            ).sum().item()

            total += y.size(0)

    val_loss /= len(
        val_loader.dataset
    )

    val_accuracy = (
        correct / total
    )

    # ==============================================
    # Print
    # ==============================================

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Accuracy: {val_accuracy:.4f}"
    )

    # ==============================================
    # Save best model
    # ==============================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "  ✓ Best model saved."
        )


print("\nTraining complete!")

print(
    f"Best model saved to: {MODEL_PATH}"
)