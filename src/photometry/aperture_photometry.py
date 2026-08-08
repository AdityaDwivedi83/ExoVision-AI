import numpy as np

def circular_aperture (frame, center, radius=2):
    """
    RETURNS A BOOLEAN MASK REPRESENTING THE CIRCULAR APERTURE
    """

    y, x = np.indices(frame.shape)

    cx, cy = center

    distance = np.sqrt((x-cx)**2 + (y-cy)**2)

    mask = distance <= radius

    return mask

def measure_flux(frame, center, radius=2):
    """
    Measure background-subtracted stellar flux.
    """

    frame = np.nan_to_num(frame)

    mask = circular_aperture(frame, center, radius)

    # Flux inside the aperture
    aperture_flux = frame[mask].sum()

    # Background pixels are everything outside the aperture
    background_pixels = frame[~mask]

    # Ignore NaNs (already converted, but this is safe)
    background_pixels = background_pixels[np.isfinite(background_pixels)]

    background_level = np.median(background_pixels)

    background_flux = background_level * mask.sum()

    net_flux = aperture_flux - background_flux

    return net_flux