# -*- coding: utf-8 -*-
"""Convert an analogue image into a photon counted image."""
from __future__ import absolute_import, division, print_function

import numpy as np


class PhotonCountException(Exception):
    """Exception class for photon_count module."""


def photon_count(e_image, thresh):
    """Convert an analog image into a photon counted image.

    Parameters
    ----------
    e_image : array_like, float
        Bias subtracted analog image (e-).
    thresh : float
        Photon counting threshold (e-). Values > thresh will be assigned a 1,
        values <= thresh will be assigned a 0.

    Returns
    -------
    pc_image : array_like, float
        Output digital image in units of photons.

    B Nemati and S Miller - UAH - 06-Aug-2018

    """
    # Check if input is an array/castable to one
    e_image = np.array(e_image).astype(float)
    if len(e_image.shape) == 0:
        raise PhotonCountException('e_image must have length > 0')

    pc_image = np.zeros(e_image.shape, dtype=int)
    pc_image[e_image > thresh] = 1

    return pc_image
