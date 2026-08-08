import numpy as np


def detect_transit_regions(flux, threshold=0.999):

    below = flux < threshold

    regions = []

    start = None

    for i, value in enumerate(below):

        if value and start is None:
            start = i

        elif not value and start is not None:
            regions.append((start, i - 1))
            start = None

    if start is not None:
        regions.append((start, len(flux) - 1))

    return regions


def extract_transit_features(flux):

    regions = detect_transit_regions(flux)

    widths = [
        end - start + 1
        for start, end in regions
    ]

    depths = [
        1.0 - np.min(flux[start:end + 1])
        for start, end in regions
    ]

    features = {

        "num_transits": len(regions),

        "max_transit_depth":
            max(depths) if depths else 0,

        "mean_transit_width":
            np.mean(widths) if widths else 0,

        "max_transit_width":
            max(widths) if widths else 0,

        "transit_fraction":
            np.sum(flux < 0.999) / len(flux)

    }

    return features