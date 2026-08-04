import matplotlib.pyplot as plt

from lightkurve import search_targetpixelfile

from src.cv.star_detector import find_star_centroid
from src.photometry.aperture_photometry import measure_flux
from src.signal_processing.lightcurve_cleaner import (
    remove_outliers,
    normalize_flux,
)

print("Downloading data...")

tpf = search_targetpixelfile(
    "Kepler-10",
    mission="Kepler"
).download()

brightness = []

for frame in tpf.flux:

    frame = frame.value

    centroid = find_star_centroid(frame)

    flux = measure_flux(frame, centroid)

    brightness.append(flux)

print("Cleaning light curve...")

clean_flux = remove_outliers(brightness)

normalized_flux = normalize_flux(clean_flux)

plt.figure(figsize=(15,5))

plt.plot(
    tpf.time.value,
    normalized_flux,
    color="navy",
    linewidth=0.7
)

plt.title("Cleaned Light Curve")

plt.xlabel("Time (BJD)")

plt.ylabel("Normalized Flux")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()