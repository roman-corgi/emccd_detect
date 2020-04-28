# -*- coding: utf-8 -*-
"""Generate cosmic hits."""
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np


def cosmic_hits(image_frame, cr_rate, frametime, pixel_pitch, max_val):
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
    max_val : float
        Maximum value of cosmic hit (e-).

    Returns
    -------
    image_frame : array_like
        Image area frame with cosmics added (e-).

    S Miller - UAH - 16-Jan-2019

    """
    # Find number of hits/frame
    nr, nc = image_frame.shape
    framesize = (nr*pixel_pitch * nc*pixel_pitch) / 10**-4  # cm^2
    hits_per_second = cr_rate * framesize
    hits_per_frame = int(round(hits_per_second * frametime))  # XXX zero case

    # Generate hit locations
    # Describe each hit as a gaussian centered at (hit_row, hit_col) and having
    # an radius of hit_rad chosen between cr_min_radius and cr_max_radius
    cr_min_radius = 0
    cr_max_radius = 2
    hit_row = np.random.uniform(low=0, high=nr-1, size=hits_per_frame)
    hit_col = np.random.uniform(low=0, high=nc-1, size=hits_per_frame)
    hit_rad = np.random.uniform(low=cr_min_radius, high=cr_max_radius,
                                size=hits_per_frame)

    # Create hits
    for i in range(hits_per_frame):
        # Get pixels where cosmic lands
        min_row = max(np.floor(hit_row[i] - hit_rad[i]).astype(int), 0)
        max_row = min(np.ceil(hit_row[i] + hit_rad[i]).astype(int), nr-1)
        min_col = max(np.floor(hit_col[i] - hit_rad[i]).astype(int), 0)
        max_col = min(np.ceil(hit_col[i] + hit_rad[i]).astype(int), nc-1)
        cols, rows = np.meshgrid(np.arange(min_col, max_col+1),
                                 np.arange(min_row, max_row+1))

        # Create gaussian
        sigma = 0.5
        a = 1 / (np.sqrt(2*np.pi) * sigma)
        b = 2 * sigma**2
        cosm_section = a * np.exp(-((rows-hit_row[i])**2 + (cols-hit_col[i])**2) / b)

        # Scale by maximum value
        cosm_section = cosm_section / np.max(cosm_section) * max_val

        # Add cosmic to frame
        image_frame[min_row:max_row+1, min_col:max_col+1] += cosm_section

    return image_frame


def sat_tails(serial_frame, full_well_serial):
    """Simulate tails created by serial register saturation.

    This is most prevalent in cosmic hits.

    Parameters
    ----------
    serial_frame : array_like
        Serial register frame (e-).
    full_well_serial : float
        Serial (gain) register full well capacity (e-).

    """
    scale = .9
    overflow = 0.
    for i, pix in enumerate(serial_frame):
        pix += overflow * scale
        serial_frame[i] = pix
        if pix > full_well_serial:
            overflow = pix - full_well_serial
        else:
            overflow = 0.
    return serial_frame


if __name__ == '__main__':
    full_well_serial = 90000

    row = np.ones(100)
    row[2] = full_well_serial * 100

    tail_row = sat_tails(row, full_well_serial)

    plt.figure()
    plt.plot(tail_row)
    plt.show()
