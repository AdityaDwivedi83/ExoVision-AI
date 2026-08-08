from src.dataset.dataset_builder import DatasetBuilder

builder = DatasetBuilder(
    "data/raw/targets.csv"
)

builder.build()