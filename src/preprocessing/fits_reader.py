from lightkurve import search_targetpixelfile
import matplotlib.pyplot as plt

print("=" * 60)
print("Downloading Target Pixel File")
print("=" * 60)

search = search_targetpixelfile(
    "Kepler-10",
    mission="Kepler"
)

tpf = search.download()

print(tpf)
print(tpf.flux.shape)

print("\nDisplaying first frame...")

tpf.plot()
plt.show()