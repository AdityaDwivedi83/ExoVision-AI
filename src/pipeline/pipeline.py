import os
import numpy as np

from lightkurve import search_targetpixelfile

from src.cv.star_detector import find_star_centroid
from src.photometry.aperture_photometry import measure_flux
from src.signal_processing.lightcurve_cleaner import (
    remove_outliers,
    normalize_flux
)


class ExoVisionPipeline:

    def __init__(self, target):

        self.target = target

        self.tpf = None
        self.time = None
        self.flux = None

    # --------------------------------------------------
    # STEP 1: Download NASA Target Pixel File
    # --------------------------------------------------

    def download_data(self):

        print(f"\nSearching MAST for {self.target}...")

        search_result = search_targetpixelfile(
            self.target,
            mission="Kepler"
        )

        if len(search_result) == 0:

            raise RuntimeError(
                f"No Target Pixel File found for {self.target}"
            )

        print(
            f"Found {len(search_result)} Target Pixel File(s)"
        )

        self.tpf = search_result[0].download()

        if self.tpf is None:

            raise RuntimeError(
                f"Could not download TPF for {self.target}"
            )

        print("Target Pixel File downloaded.")

        return self.tpf

    # --------------------------------------------------
    # STEP 2: Perform Aperture Photometry
    # --------------------------------------------------

    def extract_flux(self):

        if self.tpf is None:

            raise RuntimeError(
                "TPF has not been downloaded."
            )

        print("\nPerforming aperture photometry...")

        brightness = []

        for frame in self.tpf.flux:

            frame = frame.value

            centroid = find_star_centroid(frame)

            flux = measure_flux(
                frame,
                centroid
            )

            brightness.append(flux)

        brightness = np.asarray(brightness)

        self.time = np.asarray(
            self.tpf.time.value
        )

        self.flux = brightness

        print(
            f"Extracted {len(self.flux)} flux measurements."
        )

        return self.time, self.flux

    # --------------------------------------------------
    # STEP 3: Clean Light Curve
    # --------------------------------------------------

    def preprocess(self):

        if self.flux is None:

            raise RuntimeError(
                "Flux has not been extracted."
            )

        print("\nCleaning light curve...")

        clean_flux = remove_outliers(
            self.flux
        )

        print("Normalizing light curve...")

        normalized_flux = normalize_flux(
            clean_flux
        )

        self.flux = np.asarray(
            normalized_flux
        )

        return self.time, self.flux

    # --------------------------------------------------
    # STEP 4: Save Light Curve
    # --------------------------------------------------

    def save_light_curve(self):

        if self.time is None or self.flux is None:

            raise RuntimeError(
                "No processed light curve available."
            )

        output_dir = (
            "data/processed/light_curves"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        filename = (
            str(self.target)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

        output_path = os.path.join(
            output_dir,
            f"{filename}.npz"
        )

        np.savez(
            output_path,
            time=np.asarray(self.time),
            flux=np.asarray(self.flux)
        )

        print(
            f"\nLight curve saved to:\n"
            f"{output_path}"
        )

        return output_path

    # --------------------------------------------------
    # STEP 5: Complete Pipeline
    # --------------------------------------------------

    def run(self):

        print("\n" + "=" * 55)
        print(f"ExoVision Pipeline: {self.target}")
        print("=" * 55)

        # Download NASA data
        self.download_data()

        # Pixel frames → flux measurements
        self.extract_flux()

        # Clean + normalize
        self.preprocess()

        # Save for CNN
        self.save_light_curve()

        print("\nPipeline completed successfully!")

        return self.time, self.flux