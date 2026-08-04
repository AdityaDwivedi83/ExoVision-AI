import matplotlib.pyplot as plt

from src.signal_processing.preprocess_lightcurve import normalized_flux, tpf

from src.transit.transit_detector import (
    moving_average,
    detect_dips
)

smooth = moving_average(
    normalized_flux,
    window=25
)

dips = detect_dips(
    smooth,
    threshold=0.9995
)

plt.figure(figsize=(15,5))

plt.plot(
    tpf.time.value,
    smooth,
    color="black"
)

plt.scatter(
    tpf.time.value[dips],
    smooth[dips],
    color="red",
    s=10,
    label="Possible Transit"
)

plt.legend()

plt.show()