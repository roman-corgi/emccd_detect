# -*- coding: utf-8 -*-
import numpy as np

from cosmic_hits import cosmic_hits
from cosmic_tails import cosmic_tails
from rand_em_gain import rand_em_gain


def emccd_detect(fluxmap, exptime, gain, full_well_serial, full_well,
                 dark_rate, cic_noise, read_noise, bias, quantum_efficiency,
                 cr_rate, apply_smear=True):
    """Create an EMCCD-detected image for a given fluxmap.

    Parameters
    ----------
    fluxmap : array_like, float
        Input fluxmap (photons/pix/s).
    exptime : float
        Frame time (s).
    gain : float
        CCD gain (e-/photon).
    full_well_serial : float
        Serial register capacity (e-).
    full_well : float
        Readout register capacity (e-).
    dark_rate: float
        Dark rate (e-/pix/s).
    cic_noise : float
        Charge injection noise (e-/pix/frame).
    read_noise : float
        Read noise (e-/pix/frame).
    bias : float
        Bias offset (e-).
    quantum_efficiency : float
        Quantum efficiency.
    cr_rate : float
        Cosmic ray rate (hits/cm^2/s).
    apply_smear : bool, optional
        Apply LOWFS readout smear. Defaults to True.

    Returns
    -------
    sim_im : array_like, float
        Output simulated fluxmap.

    Notes
    -----
    The flux map must be in units of photons/pix/s. Read noise is in electrons
    and is the amplifier read noise and not the effective read noise after the
    application of EM gain. Dark current must be supplied in units of e-/pix/s,
    and cic_noise is the clock induced charge in units of e-/pix/frame.

    B Nemati and S Miller - UAH - 18-Jan-2019
    """
    fixed_pattern = np.zeros(fluxmap.shape)  # This will be modeled later

    # Mean expected electrons after inegrating over exptime
    mean_expected_e = fluxmap * exptime * quantum_efficiency

    # Mean expected dark current after integrationg over exptime
    mean_expected_dark = dark_rate * exptime
    shot_noise = mean_expected_dark + cic_noise

    # Electrons actualized at the pixels
    if apply_smear:
        expected_e = np.random.poisson(mean_expected_e
                                       + shot_noise).astype(float)
    else:
        expected_e = np.random.poisson(np.ones(fluxmap.shape)
                                       * shot_noise).astype(float)
        expected_e += mean_expected_e

    # Cosmic hits on image area
    cr_max_radius = 3  # Max radius of pixels affected by cosmic hit (pix)
    pixel_pitch = 13 * 10**-6  # Distance between pixel centers (m)
    expected_e = cosmic_hits(expected_e, cr_rate, cr_max_radius, exptime,
                             pixel_pitch, full_well)

    # Electrons capped at full well capacity of imaging area
    expected_e[expected_e > full_well] = full_well

    # Go through EM register
    expected_e_flat = expected_e.ravel(1)
    em_frame = np.zeros(fluxmap.shape)
    em_frame_flat = em_frame.ravel(1)
    indnz = expected_e.ravel(1).nonzero()[0]

    for ii in range(len(indnz)):
        ie = indnz[ii]
        em_frame_flat[ie] = rand_em_gain(expected_e_flat[ie], gain)

    expected_e = expected_e_flat.reshape(expected_e.shape, order='F')
    em_frame = em_frame_flat.reshape(em_frame.shape, order='F')

    if cr_rate:
        # Tails from cosmic hits
        em_frame = cosmic_tails(em_frame, full_well_serial, h, k, r)

    # Cap at full well capacity of gain register
    em_frame[em_frame > full_well_serial] = full_well_serial

    # Read_noise
    read_noise_map = read_noise * np.random.randn([fluxmap.shape[0],
                                                   fluxmap.shape[1]])

    sim_im = em_frame + read_noise_map + fixed_pattern + bias

    return sim_im
