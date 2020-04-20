# -*- coding: utf-8 -*-
"""Simulation for EMCCD detector."""
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.interpolate as interp

from emccd_detect.cosmic_hits import cosmic_hits
from emccd_detect.rand_em_gain import rand_em_gain


def emccd_detect(fluxmap,
                 frametime,
                 em_gain=5000.,
                 full_well_image=50000.,
                 full_well_serial=90000.,
                 dark_current=0.0028,
                 cic=0.01,
                 read_noise=100,
                 bias=0.,
                 qe=0.9,
                 cr_rate=0.,
                 pixel_pitch=13e-6,
                 shot_noise_on=True
                 ):
    """Create an EMCCD-detected image for a given fluxmap.

    Parameters
    ----------
    fluxmap : array_like, float
        Input fluxmap (photons/pix/s).
    frametime : float
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
    image_frame = image_area(fluxmap, frametime, full_well_image, dark_current,
                             cic, qe, cr_rate, pixel_pitch, shot_noise_on)

    serial_frame = serial_register(image_frame, em_gain, full_well_serial,
                                   read_noise, bias)
    return serial_frame


def image_area(fluxmap, frametime, full_well_image, dark_current, cic, qe,
               cr_rate, pixel_pitch, shot_noise_on):
    """Simulate detector image area.

    Parameters
    ----------
    fluxmap : array_like, float
        Input fluxmap (photons/pix/s).
    frametime : float
        Frame time (s).
    full_well_image : float
        Image area full well capacity (e-).
    dark_current: float
        Dark current rate (e-/pix/s).
    cic : float
        Clock induced charge (e-/pix/frame).
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
    image_frame : array_like
        Image area frame (e-).

    """
    image_frame = np.zeros([70, 70])
    ul = (0, 0)
    image_frame = embed_fluxmap(fluxmap, image_frame, ul)

    # Mean photo-electrons after inegrating over frametime
    mean_phe_map = image_frame * frametime * qe

    # Mean expected rate after integrating over frametime
    mean_dark = dark_current * frametime
    mean_noise = mean_dark + cic

    # Actualize electrons at the pixels
    if shot_noise_on:
        image_frame = np.random.poisson(mean_phe_map + mean_noise).astype(float)
    else:
        image_frame = mean_phe_map + np.random.poisson(mean_noise,
                                                       mean_phe_map.shape
                                                       ).astype(float)

    # Simulate cosmic hits on image area
    image_frame = cosmic_hits(image_frame, cr_rate, frametime, pixel_pitch,
                              full_well_image)

    # Cap at full well capacity of image area
    image_frame[image_frame > full_well_image] = full_well_image
    return image_frame


def serial_register(image_frame, em_gain, full_well_serial, read_noise, bias):
    """Simulate detector serial (gain) register.

    Parameters
    ----------
    image_frame : array_like
        Image area frame (e-).
    em_gain : float
        CCD em_gain (e-/photon).
    full_well_serial : float
        Serial (gain) register full well capacity (e-).
    read_noise : float
        Read noise (e-/pix/frame).
    bias : float
        Bias offset (e-).

    Returns
    -------
    serial_frame : array_like
        Serial register frame (e-).

    """
    # Flatten image area row by row to simulate readout to serial register
    serial_frame = image_frame.ravel()

    # Apply EM gain
    serial_frame = rand_em_gain(serial_frame, em_gain)
    # Cap at full well capacity of gain register
    serial_frame[serial_frame > full_well_serial] = full_well_serial

    # Apply fixed pattern, read noise, and bias
    serial_frame += make_fixed_pattern(serial_frame)
    serial_frame += make_read_noise(serial_frame, read_noise) + bias

    # Reshape for viewing
    return serial_frame.reshape(image_frame.shape)


def embed_fluxmap(fluxmap, image_frame, ul):
    """Add fluxmap at specified position on image section.

    Parameters
    ----------
    fluxmap : array_like
        Input fluxmap (photons/pix/s).
    image_frame : array_like
        Image area frame before electrons are actualized (photons/pix/s).
    ul : tuple
        Upper left corner of fluxmap wrt upper left corner of image section.

    Returns
    -------
    image_frame : array_like
        Image area frame before electrons are actualized (photons/pix/s).

    """
    pad = np.zeros(image_frame.shape)

    # Initially place fluxmap at 1,1 so it is padded all around
    pad[1:1+fluxmap.shape[0], 1:1+fluxmap.shape[1]] = fluxmap

    # Initialize interpolation
    rows = np.arange(pad.shape[0])
    cols = np.arange(pad.shape[1])
    f = interp.interp2d(cols, rows, pad)

    # Subtract 1 from ul coordinates to compensate for padding
    pad_interp = f(cols - (ul[1]-1), rows - (ul[0]-1))
    return image_frame + pad_interp


def make_fixed_pattern(serial_frame):
    """Simulate EMCCD fixed pattern."""
    return np.zeros(serial_frame.shape)  # This will be modeled later


def make_read_noise(serial_frame, read_noise):
    """Simulate EMCCD read noise."""
    return read_noise * np.random.normal(size=serial_frame.shape)
