# -*- coding: utf-8 -*-
"""Simulation for EMCCD detector."""
from __future__ import division, absolute_import, print_function

import numpy as np

from emccd_detect.cosmic_hits import cosmic_hits
from emccd_detect.cosmic_tails import cosmic_tails
from emccd_detect.rand_em_gain import rand_em_gain


def emccd_detect(fluxmap, exptime, em_gain, full_well_image, full_well_serial,
                 dark_current, cic, read_noise, bias, qe,
                 cr_rate, pixel_pitch, shot_noise_on=True):
    """Create an EMCCD-detected image for a given fluxmap.

    Parameters
    ----------
    fluxmap : array_like, float
        Input fluxmap (photons/pix/s).
    exptime : float
        Frame time (s).
    em_gain : float
        CCD em_gain (e-/photon).
    full_well_image : float
        Image area full well capacity (e-).
    full_well_serial : float
        Serial (gain) register full well capacity (e-).
    dark_current: float
        Dark current rate (e-/pix/s).
    cic : float
        Clock induced charge (e-/pix/frame).
    read_noise : float
        Read noise (e-/pix/frame).
    bias : float
        Bias offset (e-).
    qe : float
        Quantum efficiency.
    cr_rate : float
        Cosmic ray rate (hits/cm^2/s).
    pixel_pitch : float
        Distance between pixel centers (m).
    shot_noise_on : bool, optional
        Apply shot noise. Defaults to True.

    Returns
    -------
    serial_frame : array_like, float
        Detector output (e-).

    Notes
    -----
    Read noise is the amplifier read noise and not the effective read noise
    after the application of EM gain.

    B Nemati and S Miller - UAH - 18-Jan-2019
    """
    image_frame = image_area(fluxmap, exptime, full_well_image, dark_current,
                             cic, qe, cr_rate, pixel_pitch, shot_noise_on)

    serial_frame = serial_register(image_frame, em_gain, full_well_serial,
                                   read_noise, bias)

    return serial_frame


def image_area(fluxmap, exptime, full_well_image, dark_current, cic, qe,
               cr_rate, pixel_pitch, shot_noise_on):
    """Simulate detector image area."""
    # Mean electrons after inegrating over exptime
    mean_e = fluxmap * exptime * qe

    # Mean shot noise after integrating over exptime
    mean_dark = dark_current * exptime
    shot_noise = mean_dark + cic

    # Electrons actualized at the pixels
    if shot_noise_on:
        image_frame = np.random.poisson(mean_e + shot_noise).astype(float)
    else:
        image_frame = np.random.poisson(shot_noise,
                                        size=mean_e.shape).astype(float)
        image_frame += mean_e

    # Simulate cosmic hits on image area
    image_frame = cosmic_hits(image_frame, cr_rate, exptime, pixel_pitch,
                              full_well_image)

    # Cap electrons at full well capacity of imaging area
    image_frame[image_frame > full_well_image] = full_well_image

    return image_frame


def serial_register(image_frame, em_gain, full_well_serial, read_noise, bias):
    """Simulate detector serial (gain) register."""
    # Readout frame is flattened on a row by row basis
    image_frame_flat = image_frame.ravel()
    serial_frame_flat = np.zeros(image_frame.size)

    for i in range(len(image_frame_flat)):
        serial_frame_flat[i] = rand_em_gain(image_frame_flat[i], em_gain)

    serial_frame = serial_frame_flat.reshape(image_frame.shape)

#    if cr_rate:
#        # Tails from cosmic hits
#        serial_frame = cosmic_tails(serial_frame, full_well_serial, h, k, r)

    # Cap at full well capacity of gain register
    serial_frame[serial_frame > full_well_serial] = full_well_serial

    # Apply fixed pattern
    fixed_pattern = _generate_fixed_pattern(serial_frame)
    image_frame += fixed_pattern

    # Read_noise
    read_noise_map = read_noise * np.random.normal(size=image_frame.shape)

    serial_frame += read_noise_map + bias

    return serial_frame


def _generate_fixed_pattern(serial_frame):
    """Simulate EMCCD fixed pattern."""
    fixed_pattern = np.zeros(serial_frame.shape)  # This mat be modeled later

    return fixed_pattern
