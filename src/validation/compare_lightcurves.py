import matplotlib.pyplot as plt
import numpy as np

from src.pipeline.pipeline import ExoVisionPipeline

pipeline = ExoVisionPipeline("Kepler-10")
pipeline.run()

official = pipeline.tpf.to_lightcurve()

official_flux = official.flux.value
official_flux = official_flux / np.median(official_flux)

plt.figure(figsize=(15, 5))

plt.plot(
    pipeline.time,
    pipeline.flux,
    label="Our Pipeline",
    linewidth=1
)

plt.plot(
    official.time.value,
    official_flux,
    label="Lightkurve",
    linewidth=1,
    alpha=0.7
)

plt.legend()
plt.xlabel("Time (BJD)")
plt.ylabel("Normalized Flux")
plt.title("Pipeline Validation")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()