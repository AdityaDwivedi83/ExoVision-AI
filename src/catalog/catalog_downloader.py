import os
import requests

OUTPUT = "data/raw/koi_catalog.csv"

URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query="
    "select+kepid,"
    "kepoi_name,"
    "kepler_name,"
    "koi_disposition"
    "+from+cumulative"
    "&format=csv"
)


class CatalogDownloader:

    def download(self):

        print("Downloading NASA KOI catalog...")

        response = requests.get(URL, timeout=120)

        response.raise_for_status()

        os.makedirs("data/raw", exist_ok=True)

        with open(OUTPUT, "wb") as f:
            f.write(response.content)

        print(f"Saved to {OUTPUT}")