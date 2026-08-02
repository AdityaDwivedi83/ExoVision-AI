import numpy as np
import matplotlib.pyplot as plt
from lightkurve import search_targetpixelfile

search = search_targetpixelfile("Kepler-10", mission="Kepler")

tpf = search.download()

first_frame = tpf.flux[0].value

print("Frame Shape:", first_frame.shape)

clean_frame = np.nan_to_num(first_frame)

# Find brightest pixel
y, x = np.unravel_index(np.argmax(clean_frame), clean_frame.shape)

print(f"Detected Brightest Pixel: ({x}, {y})")

# Plot image
plt.figure(figsize=(6, 6))

plt.imshow(
    clean_frame,
    cmap="viridis",
    origin="lower"
)

# Mark detected star
plt.scatter(
    x,
    y,
    color="red",
    marker="x",
    s=200,
    label="Detected Star"
)

plt.colorbar(label="Flux (e-/s)")
plt.title("Raw Kepler Frame")
plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")
plt.legend()

plt.show()