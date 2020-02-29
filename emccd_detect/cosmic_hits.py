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
    frame_h, frame_w = frame.shape
    # Find size of frame
    framesize = frame_h*pixel_pitch * frame_w*pixel_pitch  # m^2

    # Find number of hits/frame
    hits_per_second = cr_rate * framesize/10.0**-4
    hits_per_frame = int(round(hits_per_second * frametime))

    # Generate hits
    hitsx = np.random.random([1, hits_per_frame])[0] * (frame_h-1)
    hitsy = np.random.random([1, hits_per_frame])[0] * (frame_w-1)

    # Describe each hit as a gaussian centered at (hitsx, hitsy), landing on
    # pixel (h, k), and having energies described by r (since radius is
    # proportional to energy)
    xx = np.arange(0, frame_w)
    yy = np.vstack(np.arange(frame_h-1, 0-1, -1))
    h = np.round(hitsx).astype(int)
    h[h < 1] = 1  # x (col)
    k = (frame_h-1 - np.round(hitsy)).astype(int)  # y (row)
    r = np.round(np.random.random([1, len(hitsx)])[0]*2
                 * (pixel_radius-1)+1).astype(int)

    # Create hits
    for i in range(len(hitsx)):
        # Set constants for gaussian
        sigma = r[i] / 3.75
        a = 1 / (np.sqrt(2*np.pi) * sigma)
        b = 2.0 * sigma**2
        cutoff = 0.03 * a

        rows = np.arange(max(k[i]-r[i], 0), min(k[i]+r[i], frame_h-1)+1)
        cols = np.arange(max(h[i]-r[i], 0), min(h[i]+r[i], frame_w-1)+1)
        cosm_section = a * np.exp(-((xx[cols]-hitsx[i])**2
                                  + (yy[rows]-hitsy[i])**2) / b)
        cosm_section[cosm_section <= cutoff] = 0

        # Normalize and scale by fwc_im
        cosm_section = cosm_section / np.max(cosm_section) * fwc_im
        frame[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1] += cosm_section

    return frame, h, k, r
