# -*- coding: utf-8 -*-
"""Convert an analogue image into a photon counted image."""
import numpy as np


class PhotonCountException(Exception):
    """Exception class for photon_count module."""
    pass


def photon_count(e_image, readnoise, thresh):
    """Convert an analogue image into a photon counted image.

    Parameters
    ----------
    e_image : array_like, float
        Input analague image in units of electrons.
    readnoise : float
        Detector read noise.
    thresh : float
        Photon counting threshold (to be multiplied by readnoise).

    Returns
    -------
    pc_image : array_like, float
        Output digital image in units of photons.

    B Nemati and S Miller - UAH - 06-Aug-2018
    """
    try:
        e_image_array = np.array(e_image).astype(float)
    except Exception:
        raise PhotonCountException('Error converting to numpy array')

    pc_image = np.zeros(e_image_array.shape)
    pc_image[e_image_array > (thresh * readnoise)] = 1

    return pc_image
