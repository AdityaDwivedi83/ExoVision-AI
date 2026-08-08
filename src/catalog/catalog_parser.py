import pandas as pd


class CatalogParser:

    def __init__(self, catalog_path):

        self.catalog_path = catalog_path
        self.catalog = None

    def load(self):

        self.catalog = pd.read_csv(self.catalog_path)

        print(f"Loaded {len(self.catalog)} KOIs")

        return self

    def filter(self):

        self.catalog = self.catalog[
            self.catalog["koi_disposition"].isin(
                ["CONFIRMED", "FALSE POSITIVE"]
            )
        ]

        return self

    def add_labels(self):

        self.catalog["label"] = (
            self.catalog["koi_disposition"]
            .map({
                "CONFIRMED": 1,
                "FALSE POSITIVE": 0
            })
        )

        return self

    def create_targets(self):

        confirmed = self.catalog[
            self.catalog["label"] == 1
        ].sample(
            n=50,
            random_state=42
        )

        false_positive = self.catalog[
            self.catalog["label"] == 0
        ].sample(
            n=50,
            random_state=42
        )

        sample = pd.concat(
            [confirmed, false_positive]
        ).sample(
            frac=1,
            random_state=42
        )

        targets = pd.DataFrame()

        targets["target"] = (
            "KIC " + sample["kepid"].astype(str)
        )

        targets["label"] = sample["label"].values

        return targets

    def save(self, output_path):

        targets = self.create_targets()

        targets.to_csv(
            output_path,
            index=False
        )

        print(f"Saved {len(targets)} targets.")