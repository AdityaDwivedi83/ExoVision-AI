import os

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class ExoVisionDataset(Dataset):

    def __init__(
        self,
        csv_file="data/processed/exovision_dataset.csv",
        lightcurve_dir="data/processed/light_curves",
        sequence_length=1024
    ):

        self.data = pd.read_csv(csv_file)

        self.lightcurve_dir = lightcurve_dir

        self.sequence_length = sequence_length

        print(
            f"Loaded {len(self.data)} samples."
        )

    # --------------------------------------------------
    # Resample light curve
    # --------------------------------------------------

    def resample(self, flux):

        flux = np.asarray(
            flux,
            dtype=np.float32
        )

        # Remove invalid values
        mask = np.isfinite(flux)

        flux = flux[mask]

        if len(flux) < 2:

            raise ValueError(
                "Light curve contains too few valid points."
            )

        # Original positions
        old_x = np.linspace(
            0,
            1,
            len(flux)
        )

        # New fixed positions
        new_x = np.linspace(
            0,
            1,
            self.sequence_length
        )

        # Linear interpolation
        resampled = np.interp(
            new_x,
            old_x,
            flux
        )

        return resampled.astype(
            np.float32
        )

    # --------------------------------------------------
    # Load one sample
    # --------------------------------------------------

    def __getitem__(self, index):

        row = self.data.iloc[index]

        target = row["target"]

        label = int(
            row["label"]
        )

        # Convert target into filename
        filename = (
            str(target)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

        filepath = os.path.join(
            self.lightcurve_dir,
            f"{filename}.npz"
        )

        if not os.path.exists(filepath):

            raise FileNotFoundError(
                f"Light curve not found: {filepath}"
            )

        # Load saved light curve
        data = np.load(filepath)

        flux = data["flux"]

        # Resample
        flux = self.resample(flux)

        mean = np.mean(flux)
        std = np.std(flux)

        if std > 1e-8:
            flux = (flux - mean) / std

        else:
            flux = flux - mean

        # Convert to tensor
        flux = torch.tensor(
            flux,
            dtype=torch.float32
        )

        # CNN expects:
        #
        # channels × sequence
        #
        # Therefore:
        #
        # (1024,)
        #
        # becomes:
        #
        # (1, 1024)

        flux = flux.unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return flux, label

    # --------------------------------------------------
    # Dataset length
    # --------------------------------------------------

    def __len__(self):

        return len(self.data)