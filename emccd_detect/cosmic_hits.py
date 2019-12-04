# -*- coding: utf-8 -*-
import numpy as np


def cosmic_hits(frame, matrixh, matrixw, cr_rate, frametime, pixel_pitch,
                pixel_radius, fwc_im):
    """Generate cosmic hits.

    Parameters
    ----------
    frame : :obj:`ndarray` of :obj:`float`
        Input array.
    matrixh : int
        Height of array (pix).
    matrixw : int
        Width of array (pix).
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
    frame : :obj:`ndarray` of :obj:`float`
        Output array.
    props : instance
        Class instance containing cosmic hits properties. Attributes are as
        follows:

        h : :obj:`ndarray` of :obj:`float`
            Column coordinates of cosmic hits (pix).
        k : :obj:`ndarray` of :obj:`float`
            Row coordinates of cosmic hits (pix).
        r : :obj:`ndarray` of :obj:`float`
            Radii of cosmic hits (pix).

    S Miller - UAH - 16-Jan-2019
    """
    # find size of frame
    framesize = matrixh*pixel_pitch * matrixw*pixel_pitch  # m^2

    # find number of hits/frame
    hits_per_second = cr_rate * framesize/10.0**-4
    hits_per_frame = int(round(hits_per_second * frametime))

    # generate hits
    s1 = np.random.get_state()
    hitsx = np.random.random([1, hits_per_frame])[0] * (matrixh-1)
    s2 = np.random.get_state()
    hitsy = np.random.random([1, hits_per_frame])[0] * (matrixw-1)

    # describe each hit as a gaussian centered at (hitsx, hitsy), landing on
    # pixel (h, k), and having energies described by r (since radius is
    # proportional to energy)
    xx = np.arange(0, matrixw)
    yy = np.vstack(np.arange(matrixh-1, 0-1, -1))
    h = np.round(hitsx).astype(int)
    h[h < 1] = 1  # x (col)
    k = (matrixh-1 - np.round(hitsy)).astype(int)  # y (row)
    s3 = np.random.get_state()
    r = np.round(np.random.random([1, len(hitsx)])[0]*2
                 * (pixel_radius-1)+1).astype(int)

    # create hits
    for i in range(0, len(hitsx)):
        # set constants for gaussian
        sigma = r[i] / 3.75
        a = 1 / (np.sqrt(2*np.pi) * sigma)
        b = 2.0 * sigma**2
        cutoff = 0.03 * a

        rows = np.arange(max(k[i]-r[i], 0), min(k[i]+r[i], matrixh-1)+1)
        cols = np.arange(max(h[i]-r[i], 0), min(h[i]+r[i], matrixw-1)+1)
        cosm_section = a * np.exp(-((xx[cols]-hitsx[i])**2
                                  + (yy[rows]-hitsy[i])**2) / b)
        cosm_section[cosm_section <= cutoff] = 0
        # normalize and scale by fwc_im

        cosm_section = cosm_section / np.max(cosm_section) * fwc_im

        frame[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1] += cosm_section

    class CosmicProperties:
        pass
    props = CosmicProperties()

    props.h = h
    props.k = k
    props.r = r

    return frame, props
