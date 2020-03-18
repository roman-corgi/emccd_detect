# -*- coding: utf-8 -*-
"""Generate cosmic hits."""
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np

from emccd_detect.util.imagesc import imagesc


def cosmic_hits(image_frame, cr_rate, frametime, pixel_pitch, full_well_image):
    """Generate cosmic hits.

    Parameters
    ----------
    image_frame : array_like
        Image area frame (e-).
    cr_rate : float
        Cosmic ray rate (hits/cm^2/s).
    frametime : float
        Frame time (s).
    pixel_pitch : float
        Distance between pixel centers (m).
    full_well_image : float
        Image area full well capacity (e-).

    Returns
    -------
    image_frame : array_like
        Image area frame with cosmics added (e-).

    S Miller - UAH - 16-Jan-2019

    """

    # Find number of hits/frame
    frame_r, frame_c = image_frame.shape
    framesize = (frame_r*pixel_pitch * frame_c*pixel_pitch) / 10.0**-4  # cm^2
    hits_per_second = cr_rate * framesize
    hits_per_frame = int(round(hits_per_second * frametime))  # XXX zero case

    # Generate hit locations
    # Describe each hit as a gaussian centered at (hit_row, hit_col) and having
    # an radius of hit_rad chosen between cr_min_radius and cr_max_radius
    cr_min_radius = 1
    cr_max_radius = 3
    hit_row = np.random.uniform(low=0, high=frame_r-1, size=hits_per_frame)
    hit_col = np.random.uniform(low=0, high=frame_c-1, size=hits_per_frame)
    hit_rad = np.random.uniform(low=cr_min_radius, high=cr_max_radius,
                                size=hits_per_frame)

    # Create hits
    for i in range(hits_per_frame):
        # Get pixels where cosmic lands
        min_row = max(np.ceil(hit_row[i] - hit_rad[i]).astype(int), 0)
        max_row = min(np.ceil(hit_row[i] + hit_rad[i]).astype(int), frame_r-1)
        min_col = max(np.ceil(hit_col[i] - hit_rad[i]).astype(int), 0)
        max_col = min(np.ceil(hit_col[i] + hit_rad[i]).astype(int), frame_c-1)
        cols, rows = np.meshgrid(np.arange(min_col, max_col+1),
                                 np.arange(min_row, max_row+1))

        # Create gaussian
        sigma = .5
        a = 1 / (np.sqrt(2*np.pi) * sigma)
        b = 2. * sigma**2
        cosm_section = a * np.exp(-((rows-hit_row[i])**2 + (cols-hit_col[i])**2) / b)

        # Scale
        cosm_section = cosm_section / np.max(cosm_section) * full_well_image

        image_frame[min_row:max_row+1, min_col:max_col+1] += cosm_section

    return image_frame
