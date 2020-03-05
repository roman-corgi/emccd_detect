# -*- coding: utf-8 -*-
import numpy as np


def cosmic_hits(frame, cr_rate, cr_max_radius, frametime, pixel_pitch, fwc_im):
    """Generate cosmic hits.

    Parameters
    ----------
    frame : array_like
        Input array.
    cr_rate : float
        Cosmic ray impact rate on image plane (hits/cm^2/s).
    cr_radius : float
        Maximum radius of pixels affected by one cosmic hit (pix).
    frametime : float
        Frame time (s).
    pixel_pitch : float
        Distance between pixel centers (m).
    fwc_im : float
        Full well capacity, image plane.

    Returns
    -------
    frame : array_like
        Frame with cosmics added.

    S Miller - UAH - 16-Jan-2019
    """
    # Number of frame rows and columns
    frame_r, frame_c = frame.shape

    # Find number of hits/frame
    framesize = (frame_r*pixel_pitch * frame_c*pixel_pitch) / 10.0**-4  # cm^2
    hits_per_second = cr_rate * framesize
    hits_per_frame = int(round(hits_per_second * frametime))  # XXX zero case

    # Generate hits
    # Describe each hit as a gaussian centered at (hit_col, hit_row) and having
    # an energy of hit_rad (since radius is assumed to be proportional to
    # energy)
    hit_row = np.random.uniform(low=0, high=frame_r-1, size=hits_per_frame)
    hit_col = np.random.uniform(low=0, high=frame_c-1, size=hits_per_frame)
    hit_rad = np.random.uniform(low=1, high=cr_max_radius, size=hits_per_frame)

    # Create hits
    for i in range(hits_per_frame):
        # Get pixels where cosmic lands
        min_row = max(np.round(hit_row[i] - hit_rad[i]).astype(int), 0)
        max_row = min(np.round(hit_row[i] + hit_rad[i]).astype(int), frame_r-1)
        min_col = max(np.round(hit_col[i] - hit_rad[i]).astype(int), 0)
        max_col = min(np.round(hit_col[i] + hit_rad[i]).astype(int), frame_c-1)
        cols, rows = np.meshgrid(np.arange(min_col, max_col+1),
                                 np.arange(min_row, max_row+1))

        # Create gaussian
        sigma = hit_rad[i] / 3.75
        a = 1 / (np.sqrt(2*np.pi) * sigma)
        b = 2.0 * sigma**2
        cosm_section = a * np.exp(-((rows-hit_row[i])**2 + (cols-hit_col[i])**2) / b)

        # Cut the small values of the gaussian out
        cutoff = 0.03 * a
        cosm_section[cosm_section <= cutoff] = 0

        # Scale and add cosmic
        cosm_section = cosm_section / np.max(cosm_section) * fwc_im
        frame[min_row:max_row+1, min_col:max_col+1] += cosm_section

    return frame
