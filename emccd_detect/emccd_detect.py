# -*- coding: utf-8 -*-
import numpy as np

from cosmic_hits import cosmic_hits
from cosmic_tails import cosmic_tails
from rand_em_gain import rand_em_gain


def emccd_detect(fluxmap, cr_rate, frametime, em_gain, bias, qe, fwc_im,
                 fwc_gr, dark_current, cic, read_noise, shot_noise_off=False):
    """Create an EMCCD-detected image corresponding to an input flux map.

    Parameters
    ----------
    fluxmap : array_like, float
        Input array (photons/pix/s).
    cr_rate : float
        Cosmic ray impact rate on image plane (hits/cm^2/s).
    frametime : float
        Frame time (s).
    em_gain : float
        Electromagnetic gain.
    bias : float
        Detector bias.
    qe : float
        Quantum efficiency.
    fwc_im : float
        Full well capacity, image plane.
    fwc_gr : float
        Full well capacity, gain register.
    dark_current: float
        Detector dark current.
    cic : float
        Clock induced charge.
    read_noise : float
        Detector read noise.

    Returns
    -------
    out : array_like, float
        Output array.

    Notes
    -----
    The flux map must be in units of photons per pixel per second. Read noise
    is in electrons and is the amplifier read noise and not the effective read
    noise after the application of EM gain. Dark current must be supplied in
    units of electrons per pixel per second, and CIC is the clock induced
    charge in units of e-/pix/frame.

    B Nemati and S Miller - UAH - 18-Jan-2019
    """
    matrixh, matrixw = np.shape(fluxmap)

    # Mean expected dark current after integrationg over frametime
    mean_expected_dark = dark_current * frametime

    # Mean expected electrons after inegrating over frametime
    mean_expected_e = fluxmap * frametime * qe

    fixed_pattern = np.zeros(fluxmap.shape)  # this will be modeled later

    # Electrons actualized at the pixels
    if shot_noise_off:
        expected_e = np.random.poisson(np.ones(mean_expected_e.shape) *
                                       (mean_expected_dark + cic)).astype(float)
        expected_e = expected_e + mean_expected_e
    else:
        expected_e = np.random.poisson(mean_expected_e + mean_expected_dark + cic).astype(float)  # noqa: E501

    if cr_rate:
        # Cosmic hits on image area
        pixel_pitch = 13 * 10**-6  # Distance between pixel centers (m)
        pixel_radius = 3  # Radius of pixels affected by one cosmic hit (pix)
        [expected_e, cosm_props] = cosmic_hits(expected_e, matrixh, matrixw,
                                               cr_rate, frametime,
                                               pixel_pitch, pixel_radius,
                                               fwc_im)

    # Electrons capped at full well capacity of imaging area
    expected_e[expected_e > fwc_im] = fwc_im
    expected_e_flat = expected_e.ravel(1)

    # Go through EM register
    em_frame = np.zeros([matrixh, matrixw])
    em_frame_flat = em_frame.ravel(1)
    indnz = expected_e.ravel(1).nonzero()[0]

    for ii in range(0, len(indnz)):
        ie = indnz[ii]
        em_frame_flat[ie] = rand_em_gain(expected_e_flat[ie], em_gain)

    expected_e = expected_e_flat.reshape(expected_e.shape, order='F')
    em_frame = em_frame_flat.reshape(em_frame.shape, order='F')

    if cr_rate:
        # Tails from cosmic hits
        em_frame = cosmic_tails(em_frame, matrixh, matrixw, fwc_gr,
                                cosm_props.h, cosm_props.k, cosm_props.r)

    # Cap at full well capacity of gain register
    em_frame[em_frame > fwc_gr] = fwc_gr

    # Read_noise
    read_noise_map = read_noise * np.random.standard_normal([matrixh, matrixw])

    out = em_frame + read_noise_map + fixed_pattern + bias

    return out
