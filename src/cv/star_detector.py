import numpy as np

def find_star_centroid (frame):

    frame = np.nan_to_num(frame)

    y_indices, x_indices = np.indices(frame.shape)

    total_flux = frame.sum()

    x_centroid = (x_indices * frame).sum() / total_flux
    y_centroid = (y_indices * frame).sum() / total_flux

    return x_centroid, y_centroid