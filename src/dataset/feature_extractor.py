import numpy as np
from src.transit.feature_extractor import extract_transit_features

def extract_features(time, flux):
    """
    Extract numerical features from a light curve.
    """
    features = {

        "mean_flux": np.mean(flux),

        "std_flux": np.std(flux),

        "min_flux": np.min(flux),

        "max_flux": np.max(flux),

        "median_flux": np.median(flux),

        "flux_range": np.max(flux) - np.min(flux),

        "num_points": len(flux)

    }

    transit_features = extract_transit_features(flux)

    features.update(transit_features)

    return features