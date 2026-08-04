import numpy as np


def remove_outliers(flux, sigma=5):
    """
    Remove extreme outliers using sigma clipping.
    Returns a cleaned copy of the flux array.
    """

    flux = np.asarray(flux).copy()

    median = np.median(flux)
    std = np.std(flux)

    mask = np.abs(flux - median) > sigma * std

    print(f"Removed {mask.sum()} outliers")

    flux[mask] = median

    return flux


def normalize_flux(flux):
    """
    Normalize the light curve around 1.0
    """

    flux = np.asarray(flux)

    return flux / np.median(flux)