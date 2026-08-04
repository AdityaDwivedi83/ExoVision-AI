import matplotlib.pyplot as plt
import numpy as np
from lightkurve import search_targetpixelfile

from src.cv.star_detector import find_star_centroid
from src.photometry.aperture_photometry import measure_flux

print("=" * 60)
print("Downloading Kepler Target Pixel File")
print("=" * 60)

search = search_targetpixelfile(
    "Kepler-10",
    mission="Kepler"
)

tpf = search.download()

print("Download Complete!")

brightness = []
times = []

print("\nGenerating Light Curve...\n")

for i, frame in enumerate(tpf.flux):

    frame = frame.value

    centroid = find_star_centroid(frame)

    flux = measure_flux(frame, centroid, radius=2)

    brightness.append(flux)

    times.append(tpf.time[i].value)

    if i % 500 == 0:
        print(f"Processed {i}/{len(tpf.flux)} frames")

print("\nDone!")

plt.figure(figsize=(15,5))

plt.plot(
    times,
    brightness,
    linewidth=0.6,
    color="black"
)

plt.title("Generated Light Curve - Kepler 10")
plt.xlabel("Time (BJD)")
plt.ylabel("Measured Flux")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()