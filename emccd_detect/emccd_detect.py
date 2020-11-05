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
        # Embed fluxmap in the correct position within the imaging area
        imaging_area = self.meta.embed_im(self.meta.imaging_area_zeros.copy(),
                                          'image', fluxmap)

        # Simulate the integration process
        imaging_area = self.integrate(imaging_area)

        # Simulate parallel clocking
        # Embed imaging area in full frame
        full_frame = self.meta.imaging_embed(self.meta.full_frame_zeros.copy(),
                                             imaging_area)
        pass

        # Simulate serial clocking
        full_frame_flat = self.clock_serial(full_frame)

        # Reshape from 1d to 2d
        return full_frame_flat.reshape(full_frame.shape)

    def integrate(self, imaging_area):
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

    def clock_serial(self, full_frame):
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


def make_fixed_pattern(serial_frame):
    """Simulate EMCCD fixed pattern."""
    return np.zeros(serial_frame.shape)  # This will be modeled later


def make_read_noise(serial_frame, read_noise):
    """Simulate EMCCD read noise."""
    return read_noise * np.random.normal(size=serial_frame.shape)
