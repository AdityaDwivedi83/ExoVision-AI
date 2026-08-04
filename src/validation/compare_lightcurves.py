import matplotlib.pyplot as plt
import numpy as np

from lightkurve import search_targetpixelfile

from src.cv.star_detector import find_star_centroid
from src.photometry.aperture_photometry import measure_flux

print("Downloading Target Pixel File...")

tpf = search_targetpixelfile(
    "Kepler-10",
    mission="Kepler"
).download()

print("Generating custom light curve...")

brightness = []

for frame in tpf.flux:

    frame = frame.value

    centroid = find_star_centroid(frame)

    flux = measure_flux(frame, centroid, radius=2)

    brightness.append(flux)

brightness = np.array(brightness)

print("Generating official light curve...")

official = tpf.to_lightcurve()

custom_norm = brightness / np.median(brightness)

official_flux = official.flux.value
official_norm = official_flux / np.median(official_flux)

plt.figure(figsize=(15,6))

plt.plot(
    tpf.time.value,
    custom_norm,
    label="Our Pipeline",
    linewidth=1
)

plt.plot(
    official.time.value,
    official_norm,
    label="Lightkurve",
    linewidth=1,
    alpha=0.7
)

plt.legend()

plt.xlabel("Time (BJD)")
plt.ylabel("Normalized Flux")

plt.title("Validation: Custom Photometry vs Lightkurve")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()

difference = np.abs(custom_norm - official_norm)

print("\nValidation Results")
print("-" * 40)

print("Mean Difference :", difference.mean())
print("Max Difference  :", difference.max())