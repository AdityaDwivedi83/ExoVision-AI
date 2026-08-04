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

def measure_flux (frame, center, radius=2):
    """
    Measure total flux inside the aperture
    """

    frame = np.nan_to_num(frame)

    mask = circular_aperture(frame, center, radius=2)

    total_flux = frame[mask].sum()

    return total_flux