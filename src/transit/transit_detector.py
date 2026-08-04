import numpy as np


def moving_average(signal, window=15):
    """
    Smooth the light curve.
    """

    kernel = np.ones(window) / window

    return np.convolve(signal, kernel, mode="same")


def detect_dips(flux, threshold=0.999):

    dips = np.where(flux < threshold)[0]

    return dips