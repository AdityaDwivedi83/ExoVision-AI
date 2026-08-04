import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from lightkurve import search_targetpixelfile
from src.photometry.aperture_photometry import measure_flux

search = search_targetpixelfile("Kepler-10", mission="Kepler")

tpf = search.download()

first_frame = tpf.flux[0].value

print("Frame Shape:", first_frame.shape)

clean_frame = np.nan_to_num(first_frame)

# Find brightest pixel
from src.cv.star_detector import find_star_centroid

x, y = find_star_centroid(first_frame)

print(f"Centroid = ({x:.2f}, {y:.2f})")

brightness = measure_flux(first_frame, (x, y), radius=2)
print(f"Measured Flux: {brightness:.2f}")

# Plot image
plt.figure(figsize=(6, 6))

plt.imshow(
    clean_frame,
    cmap="viridis",
    origin="lower"
)

circle = Circle(
    (x, y),
    radius=2,
    edgecolor="red",
    facecolor="none",
    linewidth=2,
    label="Photometry Aperture"
)

plt.gca().add_patch(circle)

# Mark detected star
plt.scatter(
    x,
    y,
    color="red",
    marker="x",
    s=200,
    label="Detected Star"
)

#Mark detected centroid

plt.scatter(
    x,
    y,
    color="red",
    s=200,
    marker="+",
    linewidths=3,
    label="Centroid"
)

plt.colorbar(label="Flux (e-/s)")
plt.title("Raw Kepler Frame")
plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")
plt.legend()

plt.show()