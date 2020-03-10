# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

import numpy as np

from emccd_detect.cosmic_hits import cosmic_hits
from emccd_detect.cosmic_tails import cosmic_tails
from emccd_detect.rand_em_gain import rand_em_gain


def emccd_detect(fluxmap, exptime, gain, full_well_serial, full_well,
                 dark_rate, cic_noise, read_noise, bias, quantum_efficiency,
                 cr_rate, pixel_pitch, apply_smear=True):
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
    pixel_pitch : float
        Distance between pixel centers (m).
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
    readout_frame = readout_register(fluxmap, exptime, full_well, dark_rate,
                                     cic_noise, quantum_efficiency, cr_rate,
                                     pixel_pitch, apply_smear)

    serial_frame = serial_register(readout_frame, gain, full_well_serial,
                                   read_noise, bias)

    sim_im = serial_frame

    return sim_im


def readout_register(fluxmap, exptime, full_well, dark_rate, cic_noise,
                     quantum_efficiency, cr_rate, pixel_pitch,
                     apply_smear):
    """Simulate detector readout register."""
    # Mean electrons after inegrating over exptime
    mean_e = fluxmap * exptime * quantum_efficiency

    # Mean shot noise after integrating over exptime
    mean_dark = dark_rate * exptime
    shot_noise = mean_dark + cic_noise

    # Electrons actualized at the pixels
    if apply_smear:
        readout_frame = np.random.poisson(mean_e + shot_noise).astype(float)
    else:
        readout_frame = np.random.poisson(shot_noise,
                                          size=mean_e.shape).astype(float)
        readout_frame += mean_e

    # Apply fixed pattern
    fixed_pattern = _generate_fixed_pattern(readout_frame)
    readout_frame += fixed_pattern

    # Simulate cosmic hits on image area
    readout_frame = cosmic_hits(readout_frame, cr_rate, exptime, pixel_pitch,
                                full_well)

    # Cap electrons at full well capacity of imaging area
    readout_frame[readout_frame > full_well] = full_well

    return readout_frame


def serial_register(readout_frame, gain, full_well_serial, read_noise, bias):
    """Simulate detector serial register."""
    # Readout frame is flattened on a row by row basis
    readout_frame_flat = readout_frame.ravel()
    serial_frame_flat = np.zeros(readout_frame.size)

    for i in range(len(readout_frame_flat)):
        serial_frame_flat[i] = rand_em_gain(readout_frame_flat[i], gain)

    serial_frame = serial_frame_flat.reshape(readout_frame.shape)

#    if cr_rate:
#        # Tails from cosmic hits
#        serial_frame = cosmic_tails(serial_frame, full_well_serial, h, k, r)

    # Cap at full well capacity of gain register
    serial_frame[serial_frame > full_well_serial] = full_well_serial

    # Read_noise
    read_noise_map = read_noise * np.random.normal(size=readout_frame.shape)

    serial_frame += read_noise_map + bias

    return serial_frame


def _generate_fixed_pattern(readout_frame):
    """Simulate EMCCD fixed pattern."""
    fixed_pattern = np.zeros(readout_frame.shape)  # This will be modeled later

    return fixed_pattern
