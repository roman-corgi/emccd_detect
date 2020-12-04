# -*- coding: utf-8 -*-
"""Simulation for EMCCD detector."""
from __future__ import absolute_import, division, print_function

import numpy as np

from emccd_detect.cosmics import cosmic_hits, sat_tails
from emccd_detect.rand_em_gain import rand_em_gain
from emccd_detect.util.read_metadata_wrapper import MetadataWrapper
from arcticpy.main import add_cti
from arcticpy.roe import ROE
from arcticpy.ccd import CCD
from arcticpy.traps import Trap


class EMCCDDetectException(Exception):
    """Exception class for emccd_detect module."""


class EMCCDDetectBase:
    """Base class for EMCCD detector.

    Parameters
    ----------
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
    eperdn : float
        Electrons per dn.
    shot_noise_on : bool
        Apply shot noise.

    """
    def __init__(
        self,
        em_gain,
        full_well_image,
        full_well_serial,
        dark_current,
        cic,
        read_noise,
        bias,
        qe,
        cr_rate,
        pixel_pitch,
        shot_noise_on,
        eperdn
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
        self.eperdn = eperdn
        self.shot_noise_on = shot_noise_on

        # Placeholders for trap parameters
        self.ccd = None
        self.roe = None
        self.traps = None
        self.express = None
        self.offset = None
        self.window_range = None

        # Placeholders for simulation values
        self.mean_expected_rate = None

    @property
    def eperdn(self):
        return self._eperdn

    @eperdn.setter
    def eperdn(self, eperdn):
        try:
            eperdn = float(eperdn)
        except Exception:
            raise EMCCDDetectException('eperdn value must be a float')

        if eperdn <= 0:
            raise EMCCDDetectException('eperdn value must be positve.')
        else:
            self._eperdn = eperdn

    def update_cti(
        self,
        ccd=None,
        roe=None,
        traps=None,
        express=1,
        offset=0,
        window_range=None
    ):
        # Update parameters
        self.ccd = ccd
        self.roe = roe
        self.traps = traps

        self.express = express
        self.offset = offset
        self.window_range = window_range

        # Instantiate defaults for any class instances not provided
        if self.ccd is None:
            self.ccd = CCD()
        if roe is None:
            self.roe = ROE()
        if traps is None:
            self.traps = [Trap()]

    def unset_cti(self):
        # Remove CTI simulation
        self.ccd = None
        self.roe = None
        self.traps = None

    def sim_sub_frame(self, fluxmap, frametime):
        """A fast way of adding noise to a fluxmap.

        This is a fast way of adding noise to an arbitrarily sized fluxmap.
        This method is slightly less accurate when cosmics are used, since the
        tail wrapping will be too strong. In a full frame the cosmic tails wrap
        into the next row in the prescan and trail off significantly before
        getting back to the image area, but here we only deal with the image
        so there is no prescan.

        Parameters
        ----------
        fluxmap : array_like
            Input fluxmap of arbitrary shape (phot/pix/s).
        frametime : float
            Frame exposure time (s).

        Returns
        -------
        output_counts : array_like
            Detector output counts (e-)

        """
        # Simulate the integration process
        exposed_pix_m = np.ones_like(fluxmap).astype(bool)  # No unexposed pixels
        actualized_e = self.integrate(fluxmap, frametime, exposed_pix_m)

        # Simulate parallel clocking
        parallel_counts = self.clock_parallel(actualized_e)

        # Simulate serial clocking (output will be flattened to 1d)
        empty_element_m = np.zeros_like(parallel_counts).astype(bool)  # No empty elements
        gain_counts = self.clock_serial(parallel_counts, empty_element_m)

        # Simulate amplifier and adc redout
        output_dn = self.readout(gain_counts)

        # Reshape from 1d to 2d
        return output_dn.reshape(actualized_e.shape)

    def integrate(self, fluxmap_full, frametime, exposed_pix_m):
        # Add cosmic ray effects
        # XXX Maybe change this to units of flux later
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
        # Only add CTI if update_cti has been called
        if self.ccd is not None and self.roe is not None and self.traps is not None:
            parallel_counts = add_cti(
                actualized_e,
                parallel_roe=self.roe,
                parallel_ccd=self.ccd,
                parallel_traps=self.traps,
                parallel_express=self.express,
                parallel_offset=self.offset,
                parallel_window_range=self.window_range
            )
        else:
            parallel_counts = actualized_e

        return parallel_counts

    def clock_serial(self, actualized_e_full, empty_element_m):
        # Actualize cic electrons in prescan and overscan pixels
        # XXX Another place where we are fudging a little
        actualized_e_full[empty_element_m] = np.random.poisson(actualized_e_full[empty_element_m]
                                                               + self.cic)

        # Flatten row by row
        actualized_e_full_flat = actualized_e_full.ravel()

        # Clock electrons through serial register elements
        serial_counts = self._serial_register_elements(actualized_e_full_flat)

        # Clock electrons through gain register elements
        gain_counts = self._gain_register_elements(serial_counts)

        return gain_counts

    def readout(self, gain_counts):
        # Pass electrons through amplifier
        amp_ev = self._amp(gain_counts)

        # Pass amp electron volt counts through analog to digital converter
        output_dn = self._adc(amp_ev)

        return output_dn

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

        # Set mean expected rate (commonly referred to as lambda)
        self.mean_expected_rate = mean_phe_map + mean_noise

        # Actualize electrons at the pixels
        if self.shot_noise_on:
            actualized_e = np.random.poisson(self.mean_expected_rate).astype(float)
        else:
            actualized_e = mean_phe_map + np.random.poisson(mean_noise,
                                                            mean_phe_map.shape
                                                            ).astype(float)

        # Add cosmic ray effects
        # XXX Maybe change this to units of flux later
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
        # XXX Call arcticpy here
        serial_counts = actualized_e_full_flat
        return serial_counts

    def _gain_register_elements(self, serial_counts):
        """Simulate gain register element behavior.

        Parameters
        ----------
        serial_counts : array_like
            Electrons counts after passing through serial register elements.

        Returns
        -------
        gain_counts : array_like
            Electron counts after passing through gain register elements.

        """
        # Apply EM gain
        gain_counts = np.zeros_like(serial_counts)
        gain_counts = rand_em_gain(serial_counts, self.em_gain, self.full_well_serial)

        # Simulate partial CIC for zero elements
        n_gain_elements = 604
        gain_cic = self.em_gain**(1/n_gain_elements) - 1

        # Find the zero elements
        zero_mask = (gain_counts == 0)

        # Apply CIC
        gain_counts[zero_mask] = np.random.poisson(gain_counts[zero_mask] + gain_cic)

        # Find actualized CIC counts
        cic_counts_mask = (gain_counts > 0) * zero_mask

        # Choose random starting positions for each CIC count and calculate
        # the gains based on those starting positions
        n_new = np.round(np.random.random(gain_counts[cic_counts_mask].size)
                         * (n_gain_elements-1)).astype(int)
        gains = (1 + gain_cic)**n_new

        for ind, gain in zip(cic_counts_mask.nonzero()[0], gains):
            n_in = np.array([gain_counts[ind]])
            gain_counts[ind] = rand_em_gain(n_in, gain, self.full_well_serial)


        # Simulate saturation tails
        # gain_counts = sat_tails(gain_counts, self.full_well_serial)

        # Cap at full well capacity of gain register
        gain_counts[gain_counts > self.full_well_serial] = self.full_well_serial

        return gain_counts

    def _amp(self, serial_counts):
        """Simulate amp behavior.

        Parameters
        ----------
        serial_counts : array_like
            Electron counts from the serial register.

        Returns
        -------
        amp_ev : array_like
            Output from amp (eV).

        Notes
        -----
        Read noise is the amplifier read noise and not the effective read noise
        after the application of EM gain.

        """
        # Create read noise distribution in units of electrons
        read_noise_e = self.read_noise * np.random.normal(size=serial_counts.shape)

        # Apply read noise and bias to counts to get output electron volts
        amp_ev = serial_counts + read_noise_e + self.bias

        return amp_ev

    def _adc(self, amp_ev):
        """Simulate analog to digital converter behavior.

        Parameters
        ----------
        amp_ev : array_like
            Electron volt counts from amp (eV).

        Returns
        -------
        output_dn : array_like
            Analog to digital converter output (dn).

        """
        # Convert from electron volts to dn
        output_dn = amp_ev / self.eperdn

        return output_dn


class EMCCDDetect(EMCCDDetectBase):
    """Create an EMCCD-detected image for a given fluxmap.

    This class gives a method for simulating full frames (sim_full_frame) and
    also for adding simulated noise only to the input fluxmap (sim_sub_frame).

    Parameters
    ----------
    meta_path : str
        Full path of metadata yaml.
    em_gain : float
        CCD em_gain (e-/photon). Defaults to 5000.
    full_well_image : float
        Image area full well capacity (e-). Defaults to 50000.
    full_well_serial : float
        Serial (gain) register full well capacity (e-). Defaults to None.
    dark_current: float
        Dark current rate (e-/pix/s). Defaults to 0.0028.
    cic : float
        Clock induced charge (e-/pix/frame). Defaults to 0.01.
    read_noise : float
        Read noise (e-/pix/frame). Defaults to 100.
    bias : float
        Bias offset (e-). Defaults to 0.
    qe : float
        Quantum efficiency. Defaults to 0.9.
    cr_rate : float
        Cosmic ray rate (hits/cm^2/s). Defaults to 0.
    pixel_pitch : float
        Distance between pixel centers (m). Defaults to 13e-6.
    shot_noise_on : bool
        Apply shot noise. Defaults to True.
    eperdn : float
        Electrons per dn. Defaults to None.

    """

    def __init__(
        self,
        meta_path,
        em_gain=5000.,
        full_well_image=50000.,
        full_well_serial=None,
        dark_current=0.0028,
        cic=0.01,
        read_noise=100.,
        bias=0.,
        qe=0.9,
        cr_rate=0.,
        pixel_pitch=13e-6,
        shot_noise_on=True,
        eperdn=None
    ):
        # Before inheriting base class, get metadata
        self.meta_path = meta_path
        self.meta = MetadataWrapper(self.meta_path)

        # Set defaults from metadata
        if full_well_serial is None:
            full_well_serial = self.meta.fwc
        if eperdn is None:
            eperdn = self.meta.eperdn

        super().__init__(
            em_gain=em_gain,
            full_well_image=full_well_image,
            full_well_serial=full_well_serial,
            dark_current=dark_current,
            cic=cic,
            read_noise=read_noise,
            bias=bias,
            qe=qe,
            cr_rate=cr_rate,
            pixel_pitch=pixel_pitch,
            shot_noise_on=shot_noise_on,
            eperdn=eperdn
        )

    def sim_full_frame(self, fluxmap, frametime):
        """Simulate a full detector frame.

        Note that the fluxmap provided must be the same size as the exposed
        detector pixels (labeled 'image' in metadata). A full frame including
        prescan and overscan regions will be made around the fluxmap.

        Parameters
        ----------
        fluxmap : array_like
            Input fluxmap of same shape as self.meta.geom.image (phot/pix/s).
        frametime : float
            Frame exposure time (s).

        Returns
        -------
        output_counts : array_like
            Detector output counts (dn).

        """
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
        parallel_counts = self.clock_parallel(actualized_e)

        # Initialize the serial register elements.
        full_frame_zeros = self.meta.full_frame_zeros.copy()
        # Embed the imaging area within the full frame. Create a mask for
        # referencing the prescan and overscan subsections later
        parallel_counts_full = self.meta.imaging_embed(full_frame_zeros, parallel_counts)
        empty_element_m = self.meta.mask('prescan') + self.meta.mask('overscan')
        # Simulate serial clocking
        gain_counts = self.clock_serial(parallel_counts_full, empty_element_m)

        # Simulate amplifier and adc redout
        output_dn = self.readout(gain_counts)

        # Reshape from 1d to 2d
        return output_dn.reshape(parallel_counts_full.shape)

    def slice_fluxmap(self, full_frame):
        """Return only the fluxmap portion of a full frame.

        Parameters
        ----------
        full_frame : array_like
            Simulated full frame.

        Returns
        -------
        array_like
            Fluxmap area of full frame.

        """
        return self.meta.slice_section(full_frame, 'image')

    def slice_prescan(self, full_frame):
        """Return only the prescan portion of a full frame.

        Parameters
        ----------
        full_frame : array_like
            Simulated full frame.

        Returns
        -------
        array_like
            Prescan area of a full frame.

        """
        return self.meta.slice_section(full_frame, 'prescan')


def emccd_detect(
    fluxmap,
    frametime,
    em_gain,
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

    This is a convenience function which wraps the base class implementation
    of the EMCCD simulator. It maintains the API of emccd_detect version 1.0.1.
    Note that output is in units of electrons, not dn.

    Parameters
    ----------
    fluxmap : array_like, float
        Input fluxmap (photons/pix/s).
    frametime : float
        Frame time (s).
    em_gain : float
        CCD em_gain (e-/photon).
    full_well_image : float
        Image area full well capacity (e-). Defaults to 50000.
    full_well_serial : float
        Serial (gain) register full well capacity (e-). Defaults to 90000.
    dark_current: float
        Dark current rate (e-/pix/s). Defaults to 0.0028.
    cic : float
        Clock induced charge (e-/pix/frame). Defaults to 0.01.
    read_noise : float
        Read noise (e-/pix/frame). Defaults to 100.
    bias : float
        Bias offset (e-). Defaults to 0.
    qe : float
        Quantum efficiency. Defaults to 0.9.
    cr_rate : float
        Cosmic ray rate (hits/cm^2/s). Defaults to 0.
    pixel_pitch : float
        Distance between pixel centers (m). Defaults to 13e-6.
    shot_noise_on : bool, optional
        Apply shot noise. Defaults to True.

    Returns
    -------
    serial_frame : array_like, float
        Detector output (e-).

    Notes
    -----
    The value for eperdn (electrons per dn) is hardcoded to 1. This is for
    legacy purposes, as the version 1.0.1 implementation output electrons
    instead of dn.

    B Nemati and S Miller - UAH - 18-Jan-2019

    """
    emccd = EMCCDDetectBase(
        em_gain=em_gain,
        full_well_image=full_well_image,
        full_well_serial=full_well_serial,
        dark_current=dark_current,
        cic=cic,
        read_noise=read_noise,
        bias=bias,
        qe=qe,
        cr_rate=cr_rate,
        pixel_pitch=pixel_pitch,
        shot_noise_on=shot_noise_on,
        eperdn=1.
    )

    return emccd.sim_sub_frame(fluxmap, frametime)
