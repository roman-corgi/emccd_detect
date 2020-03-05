# -*- coding: utf-8 -*-
import numpy as np

from cosmic_hits import cosmic_hits
from cosmic_tails import cosmic_tails
from rand_em_gain import rand_em_gain


def emccd_detect(frame, cr_rate, frametime, em_gain, bias, qe, fwc_im,
                 fwc_gr, dark_current, cic, read_noise, shot_noise_off=False,
                 ):
    """Create an EMCCD-detected image for a given flux map.

    Parameters
    ----------
    frame : array_like, float
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
        Detector dark current (e-/pix/s).
    cic : float
        Clock induced charge (e-/pix/frame).
    read_noise : float
        Detector read noise (e-).
    shot_noise_off : bool, optional
        Turn off detector shot noise. Defaults to False.

    Returns
    -------
    sim_im : array_like, float
        Output simulated array.

    Notes
    -----
    The flux map must be in units of photons/pix/s. Read noise is in electrons
    and is the amplifier read noise and not the effective read noise after the
    application of EM gain. Dark current must be supplied in units of e-/pix/s,
    and CIC is the clock induced charge in units of e-/pix/frame.

    B Nemati and S Miller - UAH - 18-Jan-2019
    """
    fixed_pattern = np.zeros(frame.shape)  # This will be modeled later

    # Mean expected electrons after inegrating over frametime
    mean_expected_e = frame * frametime * qe

    # Mean expected dark current after integrationg over frametime
    mean_expected_dark = dark_current * frametime
    shot_noise = mean_expected_dark + cic

    # Electrons actualized at the pixels
    if shot_noise_off:
        expected_e = np.random.poisson(np.ones(frame.shape)
                                       * shot_noise).astype(float)
        expected_e += mean_expected_e
    else:
        expected_e = np.random.poisson(mean_expected_e
                                       + shot_noise).astype(float)

    # Cosmic hits on image area
    cr_max_radius = 3  # Max radius of pixels affected by cosmic hit (pix)
    pixel_pitch = 13 * 10**-6  # Distance between pixel centers (m)
    expected_e = cosmic_hits(expected_e, cr_rate, cr_max_radius, frametime,
                             pixel_pitch, fwc_im)

    # Electrons capped at full well capacity of imaging area
    expected_e[expected_e > fwc_im] = fwc_im

    # Go through EM register
    expected_e_flat = expected_e.ravel(1)
    em_frame = np.zeros(frame.shape)
    em_frame_flat = em_frame.ravel(1)
    indnz = expected_e.ravel(1).nonzero()[0]

    for ii in range(0, len(indnz)):
        ie = indnz[ii]
        em_frame_flat[ie] = rand_em_gain(expected_e_flat[ie], em_gain)

    expected_e = expected_e_flat.reshape(expected_e.shape, order='F')
    em_frame = em_frame_flat.reshape(em_frame.shape, order='F')

    if cr_rate:
        # Tails from cosmic hits
        em_frame = cosmic_tails(em_frame, fwc_gr, h, k, r)

    # Cap at full well capacity of gain register
    em_frame[em_frame > fwc_gr] = fwc_gr

    # Read_noise
    read_noise_map = read_noise * np.random.randn([frame.shape[0],
                                                   frame.shape[1]])

    sim_im = em_frame + read_noise_map + fixed_pattern + bias

    return sim_im
