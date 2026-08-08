import os
import pandas as pd
from tqdm import tqdm

from src.pipeline.pipeline import ExoVisionPipeline
from src.dataset.feature_extractor import extract_features


class DatasetBuilder:

    def __init__(self, input_csv):

        self.input_csv = input_csv

        self.dataset = []

        self.output_file = (
            "data/processed/exovision_dataset.csv"
        )

        self.lightcurve_dir = (
            "data/processed/light_curves"
        )

        os.makedirs(
            self.lightcurve_dir,
            exist_ok=True
        )

    # --------------------------------------------------
    # Process ONE target
    # --------------------------------------------------

    def process_target(self, target, label):

        pipeline = ExoVisionPipeline(target)

        # This now:
        # 1. Downloads TPF
        # 2. Performs photometry
        # 3. Cleans light curve
        # 4. Normalizes it
        # 5. Saves .npz file

        pipeline.run()

        # Extract classical ML features
        features = extract_features(
            pipeline.time,
            pipeline.flux
        )

        features["target"] = target
        features["label"] = label

        return features

    # --------------------------------------------------
    # Build complete dataset
    # --------------------------------------------------

    def build(self):

        targets = pd.read_csv(
            self.input_csv
        )

        # Existing dataset support
        if os.path.exists(
            self.output_file
        ):

            print(
                "Existing dataset found."
            )

            existing = pd.read_csv(
                self.output_file
            )

            processed = set(
                existing["target"]
            )

            self.dataset = existing.to_dict(
                "records"
            )

        else:

            processed = set()

        failed = []

        print(
            f"\nTotal targets: {len(targets)}"
        )

        print(
            f"Already processed: {len(processed)}"
        )

        # ----------------------------------------------
        # Process targets
        # ----------------------------------------------

        for _, row in tqdm(
            targets.iterrows(),
            total=len(targets),
            desc="Processing Targets"
        ):

            target = row["target"]

            label = int(
                row["label"]
            )

            # Skip already processed targets
            if target in processed:

                continue

            try:

                result = self.process_target(
                    target,
                    label
                )

                self.dataset.append(
                    result
                )

                # Save after EVERY successful target
                pd.DataFrame(
                    self.dataset
                ).to_csv(
                    self.output_file,
                    index=False
                )

                processed.add(
                    target
                )

            except Exception as e:

                print(
                    f"\nFailed: {target}"
                )

                print(
                    f"Reason: {e}"
                )

                failed.append(
                    target
                )

        # ----------------------------------------------
        # Save failed targets
        # ----------------------------------------------

        if failed:

            pd.DataFrame(
                {
                    "target": failed
                }
            ).to_csv(
                "data/processed/failed_targets.csv",
                index=False
            )

        # ----------------------------------------------
        # Final summary
        # ----------------------------------------------

        print(
            "\n================================"
        )

        print(
            "DATASET GENERATION COMPLETE"
        )

        print(
            "================================"
        )

        print(
            f"Successful: {len(self.dataset)}"
        )

        print(
            f"Failed:     {len(failed)}"
        )

        print(
            f"\nTabular dataset:"
        )

        print(
            self.output_file
        )

        print(
            "\nLight curves:"
        )

        print(
            self.lightcurve_dir
        )