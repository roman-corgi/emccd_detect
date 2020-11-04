# -*- coding: utf-8 -*-
"""Simulation for EMCCD detector."""
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import scipy.interpolate as interp
from pathlib import Path

from emccd_detect.cosmics import cosmic_hits, sat_tails
from emccd_detect.rand_em_gain import rand_em_gain
from emccd_detect.util.read_metadata import Metadata
from emccd_detect.util.read_metadata_wrapper import MetadataWrapper

here = Path(os.path.abspath(os.path.dirname(__file__)))
META = Metadata(Path(here.parent, 'data', 'metadata.yaml'))


class EMCCDDetect:
    def __init__(self,
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
                 shot_noise_on=True,
                 meta_path=None
                 ):
        self.frametime = frametime
        self.em_gain = em_gain
        self.full_well_image = full_well_image
        self.full_well_serial = full_well_serial
        self.dark_current = dark_current
        self.cic = cic
        self.read_noise = read_noise
        self.bias = bias
        self.qe = qe
        self.cr_rate = cr_rate
        self.pixel_pitch = pixel_pitch
        self.shot_noise_on = shot_noise_on
        self.meta_path = meta_path

        # Initialize metadata
        self.meta = MetadataWrapper(self.meta_path)

    def sim_frame(self, fluxmap):
        # Embed fluxmap in the correct position within the full frame
        full_frame = self.meta.embed(self.meta.full_frame_zeros, 'image',
                                     fluxmap)

        # Get just the image counts
        image_frame = self.image_section(self.meta.imaging_slice(full_frame))

        # Get the serial counts
        full_frame = self.meta.imaging_embed(full_frame, image_frame)
        serial_frame = self.serial_register(full_frame)

        # Reshape from 1d to 2d
        return serial_frame.reshape(full_frame.shape)

    def image_section(self, imaging_area):
        # Mean photo-electrons after inegrating over frametime
        mean_phe_map = imaging_area * self.frametime * self.qe

        # Mean expected rate after integrating over frametime
        mean_dark = self.dark_current * self.frametime
        mean_noise = mean_dark + self.cic

        # Actualize electrons at the pixels
        if self.shot_noise_on:
            imaging_area = np.random.poisson(mean_phe_map + mean_noise).astype(float)
        else:
            imaging_area = mean_phe_map + np.random.poisson(mean_noise,
                                                            mean_phe_map.shape
                                                            ).astype(float)

        # Simulate cosmic hits on image section
        image = self.meta.slice_section_im(imaging_area, 'image')
        image_cosm = cosmic_hits(image, self.cr_rate, self.frametime,
                                 self.pixel_pitch, self.full_well_image)

        imaging_area = self.meta.embed_im(imaging_area, 'image', image_cosm)

        # Cap at serial full well capacity
        imaging_area[imaging_area > self.full_well_image] = self.full_well_image

        return imaging_area

    def serial_register(self, full_frame):
        # Actualize cic electrons in prescan and overscan pixels
        virtual_mask = self.meta.mask('prescan') + self.meta.mask('overscan')
        full_frame[virtual_mask] = np.random.poisson(full_frame[virtual_mask]
                                                     + self.cic)

        # Flatten image area row by row to simulate readout to serial register
        serial_frame = full_frame.ravel()

        # Apply EM gain
        serial_frame = rand_em_gain(serial_frame, self.em_gain)

        # Simulate saturation tails
        # serial_frame = sat_tails(serial_frame, full_well_serial)
        # Cap at full well capacity of gain register
        serial_frame[serial_frame > self.full_well_serial] = self.full_well_serial

        # Apply fixed pattern, read noise, and bias
        serial_frame += make_fixed_pattern(serial_frame)
        serial_frame += make_read_noise(serial_frame, self.read_noise) + self.bias

        return serial_frame


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
    # Separate image and serial register simulation into two parts, as the real
    # detector will be separated in this way
    image_frame = image_section(fluxmap, frametime, full_well_image,
                                dark_current, cic, qe, cr_rate, pixel_pitch,
                                shot_noise_on)

    serial_frame = serial_register(image_frame, em_gain, cic, full_well_serial,
                                   read_noise, bias)
    return serial_frame


def image_section(active_frame, frametime, full_well_image, dark_current, cic, qe,
                  cr_rate, pixel_pitch, shot_noise_on):
    """Simulate detector image section.

    Parameters
    ----------
    active_frame : array_like, float
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
    # Mean photo-electrons after inegrating over frametime
    mean_phe_map = active_frame * frametime * qe

    # Mean expected rate after integrating over frametime
    mean_dark = dark_current * frametime
    mean_noise = mean_dark + cic

    # Actualize electrons at the pixels
    if shot_noise_on:
        active_frame = np.random.poisson(mean_phe_map + mean_noise).astype(float)
    else:
        active_frame = mean_phe_map + np.random.poisson(mean_noise,
                                                        mean_phe_map.shape
                                                        ).astype(float)

    # Simulate cosmic hits on image area
    active_frame = cosmic_hits(active_frame, cr_rate, frametime, pixel_pitch,
                               full_well_image)

    # Create image frame
    image_frame = np.zeros([META.geom['prescan']['rows'],
                            META.geom['overscan']['cols']])

    # Place active frame in image frame
    ul = (META.geom['dark_ref_top']['rows'] + META.geom['transition_top']['rows'],
          META.geom['dark_ref_left']['cols'])
    image_frame = embed_fluxmap(active_frame, image_frame, ul)

    # Cap at full well capacity of image area
    image_frame[image_frame > full_well_image] = full_well_image
    return image_frame


def serial_register(image_frame, em_gain, cic, full_well_serial, read_noise, bias):
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
    # Make prescan
    prescan = np.zeros([META.geom['prescan']['rows'],
                        META.geom['prescan']['cols']])
    prescan = np.random.poisson(prescan + cic)

    serial_frame2d = np.append(prescan, image_frame, axis=1)

    # Flatten image area row by row to simulate readout to serial register
    serial_frame = serial_frame2d.ravel()

    # Apply EM gain
    serial_frame = rand_em_gain(serial_frame, em_gain)

    # Simulate saturation tails
    # serial_frame = sat_tails(serial_frame, full_well_serial)
    # Cap at full well capacity of gain register
    serial_frame[serial_frame > full_well_serial] = full_well_serial

    # Apply fixed pattern, read noise, and bias
    serial_frame += make_fixed_pattern(serial_frame)
    serial_frame += make_read_noise(serial_frame, read_noise) + bias

    # Reshape for viewing
    return serial_frame.reshape(serial_frame2d.shape)


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
