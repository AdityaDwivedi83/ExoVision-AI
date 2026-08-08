import torch
import torch.nn as nn


class ExoVisionCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # -------------------------
            # Block 1
            # -------------------------

            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=7,
                padding=3
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            # -------------------------
            # Block 2
            # -------------------------

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            # -------------------------
            # Block 3
            # -------------------------

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            )
        )

        # Adaptive pooling means we don't have
        # to manually calculate the flattened size.

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x