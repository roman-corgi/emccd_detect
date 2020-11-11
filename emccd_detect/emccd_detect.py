# -*- coding: utf-8 -*-
"""Simulation for EMCCD detector."""
from __future__ import absolute_import, division, print_function

import numpy as np

from emccd_detect.cosmics import cosmic_hits
from emccd_detect.rand_em_gain import rand_em_gain
from emccd_detect.util.read_metadata_wrapper import MetadataWrapper


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
                 cr_rate=1.,
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

    def sim_frame(self, fluxmap, frametime):
        # Initialize the imaging area pixels
        imaging_area_zeros = self.meta.imaging_area_zeros.copy()

        # Embed the fluxmap within the imaging area. Create a mask for
        # referencing the input fluxmap subsection later
        fluxmap_full = self.meta.embed_im(imaging_area_zeros, 'image',
                                          fluxmap)
        exposed_pix_m = self.meta.imaging_slice(self.meta.mask('image'))

        # Simulate the integration process
        actualized_e = self.integrate(fluxmap_full, frametime, exposed_pix_m)

        # Embed the imaging area within the full frame
        full_frame = self.meta.imaging_embed(self.meta.full_frame_zeros.copy(),
                                             actualized_e)

        # Simulate parallel clocking
        pass  # XXX Call arcticpy here

        # Simulate serial clocking
        full_frame_flat = self.clock_serial(full_frame)

        # Reshape from 1d to 2d
        return full_frame_flat.reshape(full_frame.shape)

    def integrate(self, fluxmap_full, frametime, exposed_pix_m):
        # Add cosmic ray effects
        # XXX Want to change this to units of flux later
        cosm_actualized_e = cosmic_hits(np.zeros_like(fluxmap_full),
                                        self.cr_rate, frametime,
                                        self.pixel_pitch, self.full_well_image)

        # Mask flux out of unexposed (covered) pixels
        fluxmap_full[exposed_pix_m == 0] = 0
        cosm_actualized_e[exposed_pix_m == 0] = 0

        # Simulate imaging area pixel effects over time
        actualized_e = self.imaging_area_elements(fluxmap_full, frametime,
                                                  cosm_actualized_e)

        return actualized_e

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

    def imaging_area_elements(self, fluxmap_full, frametime, cosm_actualized_e):
        """Simulate imaging area pixel behavior for a given fluxmap and
        frametime.

        Note that the imaging area is defined as the active pixels which are
        exposed to light plus the surrounding dark reference and transition
        areas, which are covered and recieve no light. These pixels are
        indentical to the active area, so while they recieve none of the
        fluxmap they still have the same noise profile.

        Parameters
        ----------
        fluxmap_full : array_like
            Incident photon rate fluxmap (phot/pix/s).
        frametime : float
            Frame exposure time (s).

        Returns
        -------
        actualized_e : array_like
            Map of actualized electrons (e-).

        """
        # Calculate mean photo-electrons after integrating over frametime
        mean_phe_map = fluxmap_full * self.frametime * self.qe

        # Calculate mean expected rate after integrating over frametime
        mean_dark = self.dark_current * self.frametime
        mean_noise = mean_dark + self.cic

        # Actualize electrons at the pixels
        if self.shot_noise_on:
            actualized_e = np.random.poisson(mean_phe_map
                                             + mean_noise).astype(float)
        else:
            actualized_e = mean_phe_map + np.random.poisson(mean_noise,
                                                            mean_phe_map.shape
                                                            ).astype(float)

        # Add cosmic ray effects
        # XXX Want to change this to units of flux later
        actualized_e += cosm_actualized_e

        # Cap at pixel full well capacity
        actualized_e[actualized_e > self.full_well_image] = self.full_well_image

        return actualized_e


def make_fixed_pattern(serial_frame):
    """Simulate EMCCD fixed pattern."""
    return np.zeros(serial_frame.shape)  # This will be modeled later


def make_read_noise(serial_frame, read_noise):
    """Simulate EMCCD read noise."""
    return read_noise * np.random.normal(size=serial_frame.shape)
