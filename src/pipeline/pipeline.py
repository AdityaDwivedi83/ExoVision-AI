import numpy as np
from lightkurve import search_targetpixelfile

from src.cv.star_detector import find_star_centroid
from src.photometry.aperture_photometry import measure_flux
from src.signal_processing.lightcurve_cleaner import (
    remove_outliers,
    normalize_flux,
)


class ExoVisionPipeline:

    def __init__(
        self,
        target,
        mission="Kepler",
        aperture_radius=2,
        sigma=5,
    ):

        self.target = target
        self.mission = mission
        self.aperture_radius = aperture_radius
        self.sigma = sigma

        self.tpf = None
        self.time = None
        self.flux = None
        self.centroid = None

    def download(self):

        print(f"\nDownloading {self.target} ({self.mission})...\n")

        self.tpf = (
            search_targetpixelfile(
                self.target,
                mission=self.mission
            )
            .download()
        )

        return self

    def compute_centroid(self):

        print("Computing stellar centroid...")

        first_frame = self.tpf.flux[0].value

        self.centroid = find_star_centroid(first_frame)

        print(
            f"Centroid located at "
            f"({self.centroid[0]:.2f}, {self.centroid[1]:.2f})"
        )

        return self

    def photometry(self):

        brightness = []

        print("Performing aperture photometry...")

        for frame in self.tpf.flux:

            frame = frame.value

            flux = measure_flux(
                frame,
                self.centroid,
                radius=self.aperture_radius
            )

            brightness.append(flux)

        self.time = self.tpf.time.value
        self.flux = np.array(brightness)

        return self

    def clean(self):

        print("Cleaning light curve...")

        self.flux = remove_outliers(
            self.flux,
            sigma=self.sigma
        )

        self.flux = normalize_flux(self.flux)

        return self

    def run(self):

        return (
            self.download()
                .compute_centroid()
                .photometry()
                .clean()
        )

    def summary(self):

        print("\n" + "=" * 60)
        print("ExoVision Pipeline Summary")
        print("=" * 60)

        print(f"Target           : {self.target}")
        print(f"Mission          : {self.mission}")
        print(f"Frames           : {len(self.flux)}")
        print(f"Centroid:"f"({self.centroid[0]:.2f}, {self.centroid[1]:.2f})")
        print(f"Aperture Radius  : {self.aperture_radius}")
        print(f"Sigma Threshold  : {self.sigma}")

        print(f"Time Range       : {self.time[0]:.2f} -> {self.time[-1]:.2f}")

        print(f"Mean Flux        : {self.flux.mean():.6f}")
        print(f"Minimum Flux     : {self.flux.min():.6f}")
        print(f"Maximum Flux     : {self.flux.max():.6f}")

        print("=" * 60)