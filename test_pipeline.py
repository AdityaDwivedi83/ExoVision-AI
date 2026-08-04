from src.pipeline.pipeline import ExoVisionPipeline

pipeline = ExoVisionPipeline("Kepler-10")

pipeline.run()

pipeline.summary()

print("=" * 50)
print("Pipeline Summary")
print("=" * 50)

print("Target:", pipeline.target)
print("Frames:", len(pipeline.flux))
print("Time Points:", len(pipeline.time))
print("Median Flux:", pipeline.flux.mean())