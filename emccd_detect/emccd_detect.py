# -*- coding: utf-8 -*-
"""Simulation for EMCCD detector."""
from __future__ import absolute_import, division, print_function

import numpy as np

from emccd_detect.cosmics import cosmic_hits, sat_tails
from emccd_detect.rand_em_gain import rand_em_gain
from emccd_detect.util.read_metadata_wrapper import MetadataWrapper


class EMCCDDetect:
    def __init__(self,
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

    def sim_full_frame(self, fluxmap, frametime):
        # Initialize the imaging area pixels
        imaging_area_zeros = self.meta.imaging_area_zeros.copy()
        # Embed the fluxmap within the imaging area. Create a mask for
        # referencing the input fluxmap subsection later
        fluxmap_full = self.meta.embed_im(imaging_area_zeros, 'image',
                                          fluxmap)
        exposed_pix_m = self.meta.imaging_slice(self.meta.mask('image'))

        # Simulate the integration process
        actualized_e = self.integrate(fluxmap_full, frametime, exposed_pix_m)

        # Simulate parallel clocking
        actualized_e = self.clock_parallel(actualized_e)

        # Initialize the serial register elements.
        full_frame_zeros = self.meta.full_frame_zeros.copy()
        # Embed the imaging area within the full frame. Create a mask for
        # referencing the prescan and overscan subsections later
        actualized_e_full = self.meta.imaging_embed(full_frame_zeros, actualized_e)
        empty_element_m = self.meta.mask('prescan') + self.meta.mask('overscan')

        # Simulate serial clocking
        amplified_counts = self.clock_serial(actualized_e_full, empty_element_m)

        # Reshape from 1d to 2d
        return amplified_counts.reshape(actualized_e_full.shape)

    def sim_full_frame_dn(self, fluxmap, frametime):
        return self.sim_full_frame(fluxmap, frametime) / self.meta.eperdn

    def sim_fast_frame(self, fluxmap, frametime):
        """A fast way of adding noise to a fluxmap."""
        # No unexposed pixels
        exposed_pix_m = np.ones_like(fluxmap).astype(bool)

        # Simulate the integration process
        actualized_e = self.integrate(fluxmap, frametime, exposed_pix_m)

        # Simulate parallel clocking
        actualized_e = self.clock_parallel(actualized_e)

        # No empty elements
        empty_element_m = np.zeros_like(actualized_e).astype(bool)

        # Simulate serial clocking
        amplified_counts = self.clock_serial(actualized_e, empty_element_m)

        # Reshape from 1d to 2d
        return amplified_counts.reshape(actualized_e.shape)


    def integrate(self, fluxmap_full, frametime, exposed_pix_m):
        # Add cosmic ray effects
        # XXX Want to change this to units of flux later
        cosm_actualized_e = cosmic_hits(np.zeros_like(fluxmap_full),
                                        self.cr_rate, frametime,
                                        self.pixel_pitch, self.full_well_image)

        # Mask flux out of unexposed (covered) pixels
        fluxmap_full[~exposed_pix_m] = 0
        cosm_actualized_e[~exposed_pix_m] = 0

        # Simulate imaging area pixel effects over time
        actualized_e = self._imaging_area_elements(fluxmap_full, frametime,
                                                   cosm_actualized_e)

        return actualized_e

    def clock_parallel(self, actualized_e):
        # XXX Call arcticpy here
        return actualized_e

    def clock_serial(self, actualized_e_full, empty_element_m):
        # Actualize cic electrons in prescan and overscan pixels
        # XXX Another place where we are fudging a little
        actualized_e_full[empty_element_m] = np.random.poisson(actualized_e_full[empty_element_m]
                                                               + self.cic)

        # Flatten row by row
        actualized_e_full_flat = actualized_e_full.ravel()

        # Pass electrons through serial register elements
        serial_counts = self._serial_register_elements(actualized_e_full_flat)

        # Pass electrons through amplifier
        amplified_counts = self._amp(serial_counts)

        return amplified_counts

    def _imaging_area_elements(self, fluxmap_full, frametime, cosm_actualized_e):
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
        cosm_actualized_e : array_like
            Electrons actualized from cosmic rays, same size as fluxmap_full (-e).

        Returns
        -------
        actualized_e : array_like
            Map of actualized electrons (e-).

        """
        # Calculate mean photo-electrons after integrating over frametime
        mean_phe_map = fluxmap_full * frametime * self.qe

        # Calculate mean expected rate after integrating over frametime
        mean_dark = self.dark_current * frametime
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

    def _serial_register_elements(self, actualized_e_full_flat):
        """Simulate serial register element behavior.

        Parameters
        ----------
        actualized_e_full_flat : array_like
            Electrons actualized before clocking through the serial register.

        Returns
        -------
        serial_counts : array_like
            Electrons counts after passing through serial register elements.

        """
        # Apply EM gain
        serial_counts = rand_em_gain(actualized_e_full_flat, self.em_gain)

        # Simulate saturation tails
        serial_counts = sat_tails(serial_counts, self.full_well_serial)

        # Cap at full well capacity of gain register
        serial_counts[serial_counts > self.full_well_serial] = self.full_well_serial

        return serial_counts

    def _amp(self, serial_counts):
        # Create read noise distribution
        read_noise_dist = self.read_noise * np.random.normal(size=serial_counts.shape)

        # Apply read noise and bias to counts
        return serial_counts + read_noise_dist + self.bias
