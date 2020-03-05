# -*- coding: utf-8 -*-
import numpy as np


def cosmic_hits(frame, cr_rate, frametime, pixel_pitch, pixel_radius, fwc_im):
    """Generate cosmic hits.

    Parameters
    ----------
    frame : array_like
        Input array.
    cr_rate : float
        Cosmic ray impact rate on image plane (hits/cm^2/s).
    frametime : float
        Frame time (s).
    pixel_pitch : float
        Distance between pixel centers (m).
    pixel_radius : float
        Radius of pixels affected by one cosmic hit (pix).
    fwc_im : float
        Full well capacity, image plane.

    Returns
    -------
    frame : array_like
        Output array.
    h : array_like
        Column coordinates of cosmic hits (pix).
    k : array_like
        Row coordinates of cosmic hits (pix).
    r : array_like
        Radii of cosmic hits (pix).

    S Miller - UAH - 16-Jan-2019
    """
    # Frame number of rows and columns
    frame_r, frame_c = frame.shape

    # Find number of hits/frame
    framesize = (frame_r*pixel_pitch * frame_c*pixel_pitch) / 10.0**-4  # cm^2
    hits_per_second = cr_rate * framesize
    hits_per_frame = int(round(hits_per_second * frametime))  # XXX zero case

    # Generate hits
    hit_row = np.random.uniform(low=0, high=frame_r-1, size=hits_per_frame)
    hit_col = np.random.uniform(low=0, high=frame_c-1, size=hits_per_frame)

    # Describe each hit as a gaussian centered at (hit_col, hit_row), landing on
    # pixel (h, k), and having energies described by r (since radius is
    # proportional to energy)
    h = np.round(hit_col).astype(int)
    k = np.round(hit_row).astype(int)
    r = np.round(np.random.random_integers(hits_per_frame)*2 * (pixel_radius-1)+1).astype(int)

    # Create hits
    for i in range(hits_per_frame):
        # Set constants for gaussian
        sigma = r[i] / 3.75
        a = 1 / (np.sqrt(2*np.pi) * sigma)
        b = 2.0 * sigma**2
        cutoff = 0.03 * a

        rows = np.arange(max(k[i]-r[i], 0), min(k[i]+r[i], frame_r-1)+1)
        cols = np.arange(max(h[i]-r[i], 0), min(h[i]+r[i], frame_c-1)+1)
        cosm_section = a * np.exp(-((cols-hit_col[i])**2 + (rows-hit_row[i])**2) / b)
        cosm_section[cosm_section <= cutoff] = 0

        # Normalize and scale by fwc_im
        cosm_section = cosm_section / np.max(cosm_section) * fwc_im
        frame[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1] += cosm_section

    return frame, h, k, r
