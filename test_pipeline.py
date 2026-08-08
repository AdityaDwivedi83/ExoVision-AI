from src.pipeline.pipeline import ExoVisionPipeline


pipeline = ExoVisionPipeline(
    "KIC 7350067"
)

time, flux = pipeline.run()

print("\n----------------------------")
print("TEST RESULTS")
print("----------------------------")

print("Time shape:", time.shape)
print("Flux shape:", flux.shape)

print("First 10 flux values:")
print(flux[:10])