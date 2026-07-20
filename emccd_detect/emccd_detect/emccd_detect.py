# -*- coding: utf-8 -*-
"""Simulation for EMCCD detector."""

import os
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits

from emccd_detect.cosmics import cosmic_hits, sat_tails
from emccd_detect.rand_em_gain import rand_em_gain, partial_CIC
from emccd_detect.nonlinearity import apply_relgains
from emccd_detect.util.read_metadata_wrapper import MetadataWrapper
try:
    from arcticpy import add_cti, CCD, ROE, TrapInstantCapture
except:
    pass


# NOTE Roman's EXCAM EMCCD has a field-free region of 8um thick, so pre-gain-register saturation
# is unlikely to cause vertical blooming during parallel clocking, and cosmic rays will have a 
# spread-out hit.  However, if pixel is saturated before gain register, vertical blooming 
# will occur.  Charge diffusion will also have a spreading effect on PSFs.  The input flux map
# is assumed to be the signal that hits after diffusion has occurred.  

# NOTE In general, we try to keep e- values as float instead of int, even though there is no such 
# thing as a fraction of an electron, since EM gain,
# k gain, nonlinearity, master flat, etc are calibrated assuming fractions of electrons, and 
# we can get fractions of electrons for general gain values.  So we just round 
# DN output to integer at the end.

# NOTE The EM gain register is designed so that charge beyond the serial FWC spills
# into neighboring pixels downstream.  In addition, high EM gain can cause surface traps in 
# the gain register, which increases the length of serial tails further.  These effects are simulated.  
# Partial CIC is observationally degenerate with the effect of surface traps which release a captured trap
# many pixels downstream.  Probability of capture increases with charge packeet (probability ~1 for ~100e- or higher).
# Small effect for low to moderate EM gains.  

# NOTE A value of Q=1e-3 was found using maximum likelihood estimation from real 
# Roman testbed data at a commanded gain of 5000, which serves as a benchmark value. However,
# the model used for the fit used an approximate form for the partial CIC probability, out of computational necessity, 
# which used an average gain stage value which was representative of all the stages, 
# so the fitted value of 1e-3 is physically akin to the middle stages having that value.  It was verified 
# that this is similar to P/40 for all the stages. 

class EMCCDDetectException(Exception):
    """Exception class for emccd_detect module."""


class EMCCDDetectBase:
    """Base class for EMCCD detector.

    Parameters
    ----------
    em_gain : float
        Electron multiplying gain (e-/photoelectron).
    full_well_image : float
        Image area full well capacity (e-).
    full_well_serial : float
        Serial (gain) register full well capacity (e-).
    dark_current: float
        Dark current rate (e-/pix/s).
    cic : float
        Clock induced charge (CIC) (e-/pix/frame).
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
    nbits : int
        Number of bits used by the ADC readout. Must be between 1 and 64,
        inclusive.
    numel_gain_register : int
        Number of gain register elements. 
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
        eperdn,
        nbits,
        numel_gain_register,
        **kwargs
    ):
        # Input checks
        if not isinstance(nbits, (int, np.integer)):
            raise EMCCDDetectException('nbits must be an integer')
        if nbits < 1 or nbits > 64:
            raise EMCCDDetectException('nbits must be between 1 and 64, '
                                       'inclusive')

        self.em_gain = em_gain
        self.full_well_image = full_well_image
        self.full_well_serial = full_well_serial
        self.dark_current = dark_current
        self.cic = cic
        self.read_noise = read_noise
        self.bias = bias
        if self.bias < self.read_noise:
            warnings.warn('Bias is less than the read noise, so clipping of negative values may occur. Bias is usually at least a few times the read noise.')
        self.qe = qe
        self.cr_rate = cr_rate
        self.pixel_pitch = pixel_pitch
        self.eperdn = eperdn
        self.nbits = nbits
        self.numel_gain_register = numel_gain_register


        # Placeholders for trap parameters
        self.parallel_ccd = None
        self.parallel_roe = None
        self.parallel_traps = None
        self.parallel_express = None
        self.serial_ccd = None
        self.serial_roe = None
        self.serial_traps = None
        self.serial_express = None


        # Placeholders for derived values
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
            raise EMCCDDetectException('eperdn values must be positve.')
        else:
            self._eperdn = eperdn

    try:
        def update_cti(
            self,
            parallel_ccd=None,
            parallel_roe=None,
            parallel_traps=None,
            parallel_express=1,
            serial_ccd=None,
            serial_roe=None,
            serial_traps=None,
            serial_express=1,
            parallel=True,
            serial=True,
            **kwargs # any other arguments that arcticpy.add_cti() might accept
        ):
            '''See arcticpy documentation for details on parameters. Any arguments
            not explicitly listed here can be handed to arcticpy.add_cti() via
            kwargs.

            Parallel and serial CTI can each be switched on or off via the
            "parallel" and "serial" arguments of this function.  True means that
            type of CTI is simulated.  Both are True by default.'''
            # Update parameters
            self.parallel_ccd = parallel_ccd
            self.parallel_roe = parallel_roe
            self.parallel_traps = parallel_traps
            self.parallel_express = parallel_express
            self.serial_ccd = serial_ccd
            self.serial_roe = serial_roe
            self.serial_traps = serial_traps
            self.serial_express = serial_express
            self.kwargs = kwargs
            self.parallel = parallel
            self.serial = serial

            # Instantiate defaults for any class instances not provided

            if parallel_ccd is None:
                self.parallel_ccd = CCD()
            if parallel_roe is None:
                self.parallel_roe = ROE()
            if parallel_traps is None:
                #self.traps = [Trap()]
                self.parallel_traps = [TrapInstantCapture()]
            if self.parallel is False: # overrides
                self.parallel_ccd = None
                self.parallel_roe = None
                self.parallel_traps = None

            if serial_ccd is None:
                self.serial_ccd = CCD()
            if serial_roe is None:
                self.serial_roe = ROE()
            if serial_traps is None:
                self.serial_traps = [TrapInstantCapture()]
            if self.serial is False: #overrides
                self.serial_ccd = None
                self.serial_roe = None
                self.serial_traps = None

        def unset_cti(self):
            '''This turns off all CTI implementation.'''
            # Remove CTI simulation
            self.parallel_ccd = None
            self.parallel_roe = None
            self.parallel_traps = None
            self.serial_ccd = None
            self.serial_roe = None
            self.serial_traps = None
    except:
        pass

    def sim_sub_frame(self, fluxmap, frametime):
        """Simulate a partial detector frame.

        This runs the same algorithm as sim_full_frame, but only on the given
        fluxmap without surrounding it with prescan/overscan. The input fluxmap
        array may be arbitrary in shape and an image array of the same shape
        will be returned.

        Parameters
        ----------
        fluxmap : array_like
            Input fluxmap of arbitrary shape (phot/pix/s).
        frametime : float
            Frame exposure time (s).

        Returns
        -------
        output_counts : array_like
            Detector output counts, same shape as input fluxmap (dn).

        Notes
        -----
        This method is just as accurate and will return the same results as if
        the user ran sim_full_frame and then subsectioned the input fluxmap,
        with the exception of cosmic tails.

        It is slightly less accurate when cosmics are used, since the tail
        wrapping will be too strong. In a full frame the cosmic tails wrap into
        the next row in the prescan and trail off significantly before getting
        back to the image area, but here there is no prescan so the tails will
        be immediately wrapped back into the image.

        """
        # Simulate the integration process
        exposed_pix_m = np.ones_like(fluxmap).astype(bool)  # No unexposed pixels
        actualized_e = self.integrate(fluxmap.copy(), frametime, exposed_pix_m)

        # Simulate parallel clocking
        parallel_counts = self.clock_parallel(actualized_e)

        # Simulate serial clocking (output will be flattened to 1d)
        empty_element_m = np.zeros_like(parallel_counts).astype(bool)  # No empty elements
        gain_counts = self.clock_serial(parallel_counts, empty_element_m)

        # Cap at full well capacity of gain register for sim_sub_frame() since excess charge would presumably get cleaned out or placed onto 
        # the next frame, which is not simulated here. 
        gain_counts[gain_counts > self.full_well_serial] = self.full_well_serial

        # Simulate amplifier and adc redout
        output_dn = self.readout(gain_counts.reshape(parallel_counts.shape), frametime)

        # Reshape from 1d to 2d
        return output_dn.reshape(actualized_e.shape)

    def integrate(self, fluxmap_full, frametime, exposed_pix_m):
        # Add cosmic ray effects
        # inputs to EMCCDDetect but not EMCCDDetectBase:
        if not hasattr(self, 'zff'):
            self.zff = 8e-6 
        if not hasattr(self, 'loc'):
            self.loc = 1590
        if not hasattr(self, 'scale'):
            self.scale = 550
        if not hasattr(self, 'oversample_factor'):
            self.oversample_factor = 10
        cosm_actualized_e = cosmic_hits(np.zeros_like(fluxmap_full),
                                        self.cr_rate, frametime,
                                        self.pixel_pitch, self.zff, self.loc,
                                        self.scale, self.oversample_factor)
        
        # Mask flux out of unexposed (covered) pixels
        fluxmap_full[~exposed_pix_m] = 0
        cosm_actualized_e[~exposed_pix_m] = 0

        # Simulate imaging area pixel effects over time
        actualized_e = self._imaging_area_elements(fluxmap_full, frametime,
                                                   cosm_actualized_e)

        return actualized_e

    def clock_parallel(self, actualized_e):

        if not hasattr(self, 'upstream_spill_prob'):
            self.upstream_spill_prob = None #no overspill in this case (perhaps very good overspill protection in place for the detector)
        if self.upstream_spill_prob is not None:
            # account for clock-induced transfer overflow (and effectively covers continuous blooming apart from clocking)
            sat_rows, sat_cols = np.where(actualized_e > self.full_well_image)
            clock_iteration = 0
            # NOTE each row gets read serially, along with prescan and serial overscan 
            # before and after (blank serial clockings).  Then after last row read 
            # serially, extra blank "rows" (with their own pre- and overscan) are 
            # clocked.  So no vertical overspill into overscan here; extra leftover 
            # saturation after the blooming process gets handled in either serial 
            # register overspill or sat_tails(), which in effect handles overspill in 
            # gain register.
            while sat_rows.size > 0 and clock_iteration < actualized_e.shape[0]:
                # saturated pixels most likely to overspill upstream, away from read-out direction, or 0-row direction in Roman CGI EMCCDs.
                # Upstream: going toward higher rows, away from read-out direction
                # so let p > 0.5 influence bias in that direction.  Below returns 0 or 1. 
                spill_pixels = np.random.binomial(n=np.ones(len(sat_rows)).astype(int), p=self.upstream_spill_prob)
                spill_pixels[spill_pixels==0] = -1 # small chance of spilling downstream
                # charge that was planned to move downstream in this case just stays put since it can't go downstream as the row is in the serial register now; 
                # charge can overspill while in serial register later.
                # use a single boolean mask to avoid chained indexing which won't modify the original array
                mask_zero_downstream = (sat_rows == 0) & (spill_pixels == -1)
                spill_pixels[mask_zero_downstream] = 0
                # last row: charge could flow downstream, which would be into overscan region, which can handle it 
                spillover_rows = sat_rows + spill_pixels
                beyond_last_row = np.where(spillover_rows >= actualized_e.shape[0])[0] 
                spillover_rows[beyond_last_row] += -1 # charge can't go anywhere physically beyond last row, so it sits in last row instead for this iteration

                spill_amount = actualized_e[sat_rows, sat_cols].copy() - self.full_well_image
                # If these spillover coordinates are not all unique, then not all the locations get the addition.  Have to handle carefully:
                # Use np.add.at to accumulate into possibly repeated indices without an explicit Python loop.
                np.add.at(actualized_e, (spillover_rows, sat_cols), spill_amount)
                actualized_e[sat_rows, sat_cols] -= spill_amount
                sat_rows, sat_cols = np.where(actualized_e  > self.full_well_image)
                clock_iteration += 1

        # Physically, some pixels in principle can arrive at serial register still saturated.
        # Physically, CTI and clock-induced transfer overflow should happen together, but this order here is good enough (no hack of arCTIc required)

        # Only add CTI if update_cti has been called
        if self.parallel_ccd is not None and self.parallel_roe is not None and self.parallel_traps is not None:
            try:
                parallel_counts = add_cti(
                    actualized_e.copy(),
                    parallel_roe=self.parallel_roe,
                    parallel_ccd=self.parallel_ccd,
                    parallel_traps=self.parallel_traps,
                    parallel_express=self.parallel_express,
                    **self.kwargs
                )
            except:
                parallel_counts = add_cti(
                    actualized_e.copy(),
                    parallel_roe=self.parallel_roe,
                    parallel_ccd=self.parallel_ccd,
                    parallel_traps=self.parallel_traps,
                    parallel_express=self.parallel_express,
                    parallel_window_range=0,
                    **self.kwargs
                )
        else:
            parallel_counts = actualized_e

        return parallel_counts

    def clock_serial(self, actualized_e_full, empty_element_m):
        # Actualize cic electrons in prescan and overscan pixels
        # Another place where we are fudging a little as far as the order of operations(?)
        actualized_e_full[empty_element_m] += np.random.poisson(actualized_e_full[empty_element_m]
                                                               + self.cic)
        #XXX Could treat overspill in serial register here, separately from sat_tails() 
        # which effectively handles overspill in gain register (and maybe some/all of the effect of serial register)?

        # add serial CTI; the addition of CIC (serial and parallel) and smear is really 
        # *during* the addition of CTI, but this corrective effect would not be very significant 
        if self.serial_ccd is not None and self.serial_roe is not None and self.serial_traps is not None:
            try:
                cti_actualized_e_full = add_cti(
                        actualized_e_full.copy(),
                        serial_roe=self.serial_roe,
                        serial_ccd=self.serial_ccd,
                        serial_traps=self.serial_traps,
                        serial_express=self.serial_express,
                        **self.kwargs
                    )
            except:
                cti_actualized_e_full = add_cti(
                        actualized_e_full.copy(),
                        serial_roe=self.serial_roe,
                        serial_ccd=self.serial_ccd,
                        serial_traps=self.serial_traps,
                        serial_express=self.serial_express,
                        serial_window_range=0,
                        **self.kwargs
                    )
        else:
            cti_actualized_e_full = actualized_e_full

        # Flatten row by row
        actualized_e_full_flat = cti_actualized_e_full.ravel()

        # Clock electrons through serial register elements
        serial_counts = self._serial_register_elements(actualized_e_full_flat)

        # Clock electrons through gain register elements
        gain_counts = self._gain_register_elements(serial_counts)

        return gain_counts

    def readout(self, gain_counts, frametime):
        # Pass electrons through amplifier
        amp_ev = self._amp(gain_counts, frametime)

        # Pass amp electron volt counts through analog to digital converter,
        # applying nonlinearity if applicable
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
            Electrons actualized from cosmic rays, same size as fluxmap_full (e-).

        Returns
        -------
        actualized_e : array_like
            Map of actualized electrons (e-).

        """
        # Calculate mean photo-electrons after integrating over frametime
        mean_phe_map = fluxmap_full * frametime  
        
        # credit for this smearing code: Peter Williams, Tellus1, 2024
        # XXX Technically, smearing adds electrons to each pixel during 
        # parallel clocking, which increases the chance of charge capture 
        # for CTI, but simulating this small effect would require 
        # hacking arCTIc.  
        # Also, smearing is from flux and thus subject to pixel response non-uniformity, 
        # so master flat implemented after smear
        if hasattr(self, 'row_read_time'):
            smear = np.zeros_like(fluxmap_full)
            m = len(smear)
            for r in range(m):
                columnsum = 0
                for i in range(r+1):
                    columnsum = columnsum + self.row_read_time*fluxmap_full[i,:]
                smear[r,:] = columnsum
            # add in effect of smear
            mean_phe_map = mean_phe_map + smear 

        # apply non-uniformity of pixel responsivity via master flat; includes effect of dead/poor-response pixels
        if hasattr(self, 'flat_path'): # then self.meta should exist as well
            if self.flat_path is not None:
                with fits.open(self.flat_path) as hdul:
                    self.flat = hdul[1].data
                if (self.flat < 0).any():
                    raise EMCCDDetectException('Master flat must not contain '
                                                'negative values.')
                if self.flat.shape != fluxmap_full.shape:
                    imaging_area_ones = np.ones_like(fluxmap_full)
                    # Attempt to embed the flat within the 
                    # imaging+shielded area
                    self.flat_im = self.meta.embed_im(imaging_area_ones, 
                                                    'image', self.flat.copy())
                    if self.flat_im.shape != fluxmap_full.shape:
                        raise EMCCDDetectException('Master flat shape must '
                                                'agree with shape of fluxmap.')
                    else:
                        mean_phe_map *= self.flat_im
                else:   
                    mean_phe_map *= self.flat
        
        #apply QE
        mean_phe_map = mean_phe_map * self.qe

        if not hasattr(self, 'hot_pixel_path'):
            self.hot_pixel_path = None
        # Calculate mean expected rate after integrating over frametime
        if self.hot_pixel_path is not None:
            with fits.open(self.hot_pixel_path) as hdul:
                self.hot_pixel_map = hdul[1].data
            if (self.hot_pixel_map < 0).any():
                raise EMCCDDetectException('Hot pixel map must not contain '
                                            'negative values.')
            if self.hot_pixel_map.shape != fluxmap_full.shape:
                imaging_area_ones = np.ones_like(fluxmap_full)
                # Attempt to embed the hot pixel map within the 
                # imaging+shielded area
                self.hot_pixel_map_im = self.meta.embed_im(imaging_area_ones, 
                                                'image', self.hot_pixel_map.copy())
                if self.hot_pixel_map_im.shape != fluxmap_full.shape:
                    raise EMCCDDetectException('Hot pixel map shape must '
                                            'agree with shape of fluxmap.')
                else:
                    mean_dark = self.dark_current * frametime * self.hot_pixel_map_im
            else:   
                mean_dark = self.dark_current * frametime * self.hot_pixel_map
        else:
            mean_dark = self.dark_current * frametime
        mean_noise = mean_dark + self.cic

        # Set mean expected rate (commonly referred to as lambda)
        self.mean_expected_rate = mean_phe_map + mean_noise

        # Poisson process of individual photons hitting detector; CIC and dark current also 
        # governed by Poisson process.
        # Actualize electrons at the pixels 
        actualized_e = np.random.poisson(self.mean_expected_rate).astype(float)

        # Add cosmic ray effects
        actualized_e += cosm_actualized_e

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
        # NOTE To be fully accurate, the pixel values should be marched through the gain register sequentially
        # since spillover to neighboring gain stages affects the multiplication (and trap capture).  But this 
        # would mean the simulation process would take numel_gain_register times longer; we just simulate all 
        # pixels at once, each marching through the gain register, and the spillover and trap release in the register 
        # is effectively performed with sat_tails() afterwards. 
        gain_counts = np.zeros_like(serial_counts)
        if not hasattr(self, 'fast_gain_mode'):
            self.fast_gain_mode = False # use the fully accurate method
        if not hasattr(self, 'gain_stage_specs'):
            self.gain_stage_specs = None # default value
        all_gain_stage_specs = {n: self.em_gain**(1/self.numel_gain_register) - 1 for n in range(1, self.numel_gain_register+1)}
        if self.gain_stage_specs is not None:
            for num_stages_left in range(1, self.numel_gain_register+1):
                if num_stages_left in self.gain_stage_specs.keys():
                    all_gain_stage_specs[num_stages_left] = self.gain_stage_specs[num_stages_left]
        gain_counts = rand_em_gain(
            n_in_array=serial_counts,
            gain_stage_specs=all_gain_stage_specs, 
            numel_gain_register=self.numel_gain_register, fast_gain_mode=self.fast_gain_mode
            )

        # assuming partial CIC independently propagates through gain register 
        #if hasattr() for the inputs in emccdDetect but not base class
        if hasattr(self, 'gain_CIC_Q') and hasattr(self, 'gain_CIC_specs'):
            if self.gain_CIC_Q != 0 or self.gain_CIC_specs is not None:
                partial_cic = partial_CIC(gain_counts.size, 
                                        all_gain_stage_specs,
                                        self.numel_gain_register, 
                                        self.gain_CIC_Q,
                                        self.gain_CIC_specs)
                gain_counts = gain_counts + partial_cic
    
        # Simulate saturation tails
        # Cosmic tails mainly due to downstream spillover in gain register and trapping effects in gain register; 
        # not modeled with arCTIc, where serial register is merely clocking with
        # no EM gain(?), so we simulate these effects here.
        # The traps in the gain register are simulated in sat_tails here.
        if not hasattr(self, 'tail_length'):
            self.tail_length = int(np.round(self.em_gain*40/1000)) # default value
        gain_counts_tails = sat_tails(gain_counts, self.full_well_serial, self.tail_length)

        return gain_counts_tails

    def _amp(self, serial_counts, frametime):
        """Simulate amp behavior.

        Parameters
        ----------
        serial_counts : array_like
            Electron counts from the serial register.
        frametime: float
            Frame exposure time (s).

        Returns
        -------
        amp_ev : array_like
            Output from amp (eV).

        Notes
        -----
        Read noise is the amplifier read noise and not the effective read noise
        after the application of EM gain.

        """
        # Create read noise distribution in units of electrons XXX add in optional mu for mean?
        
        # get pseudo-random state to enable consistent FPN simulation given the 
        # same settings
        old_state = np.random.get_state()

        if hasattr(self, 'fpn_path'):
            here = os.path.abspath(os.path.dirname(__file__))
            if self.fpn_path is not None and self.fpn_path != 'roman':
                with fits.open(self.fpn_path) as hdul:
                    self.fpn = hdul[0].data
            elif self.fpn_path == "roman":
                if serial_counts.shape[0] == self.meta.data['frame_rows'] and serial_counts.shape[1] == self.meta.data['frame_cols']:
                    with fits.open(Path(here, 'util', 'fpn_map.fits')) as hdul:
                        frame_data = hdul[0].data
                    self.fpn = frame_data
                elif serial_counts.shape[0] <= self.meta.data['geom']['image']['rows'] and serial_counts.shape[1] <= self.meta.data['geom']['image']['cols']:
                    #If the area specificed by the user is not full frame then
                    #this code cuts out a portion/the entire image area on the fpn map to apply to amp
                    with fits.open(Path(here, 'util', 'fpn_map.fits')) as hdul:
                        frame_data = hdul[0].data
                    r0c0 = self.meta.data['geom']['image']['r0c0']
                    subframe_array = frame_data[r0c0[0]:r0c0[0]+serial_counts.shape[0], r0c0[1]:r0c0[1]+serial_counts.shape[1]]
                    self.fpn = subframe_array
                else:
                    raise EMCCDDetectException('Input fpn_path specifies an array that is incompatible with Roman full frame and Roman image/sub-image area.  If '
                                               'Roman FPN pattern desired for a different frame shape, please specify your own custom filepath.')
            elif self.fpn_path is None:
                seed2 = self.em_gain + self.bias + frametime + self.dark_current
                np.random.seed(int(seed2))
                self.fpn = np.zeros_like(serial_counts)
                bias_row_offset = np.random.normal(0, self.bias_sigma_row, self.fpn.shape[0])
                self.fpn += bias_row_offset[:, np.newaxis]
                bias_col_offset = np.random.normal(0, self.bias_sigma_col, self.fpn.shape[1])
                self.fpn += bias_col_offset[np.newaxis, :]
            else: #exception
                raise EMCCDDetectException('Current input for fpn_path should be a string file path, \'roman\', or None.')
        else: # just self.bias constant
            self.fpn = np.zeros_like(serial_counts)

        np.random.set_state(old_state)

        # Create read noise distribution in units of electrons
        read_noise_e = self.read_noise * np.random.normal(size=serial_counts.shape)

        # Apply read noise and bias to counts to get output electron volts
        amp_ev = serial_counts + read_noise_e + self.fpn + self.bias

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
        # Convert from electron volts to dn and apply nonlin if applicable
        dn = amp_ev / self.eperdn
        if hasattr(self, 'nonlin_path'):
            if self.nonlin_path is not None:
                nonlin_factors = apply_relgains(dn, self.em_gain,
                                                self.nonlin_path)
                dn *= nonlin_factors
        dn_min = 0
        dn_max = 2**self.nbits - 1
        output_dn = np.clip(dn, dn_min, dn_max).astype(np.uint64)

        return output_dn


class EMCCDDetect(EMCCDDetectBase):
    """Create an EMCCD-detected image for a given fluxmap.

    This class gives a method for simulating full frames (sim_full_frame) and
    also for adding simulated noise only to the input fluxmap (sim_sub_frame).

    Parameters
    ----------
    em_gain : float
        Electron multiplying gain (e-/photoelectron). Defaults to 1.
    full_well_image : float
        Image area full well capacity (e-). Defaults to 78000.
    full_well_serial : float
        Serial (gain) register full well capacity (e-). Defaults to 105000.
    dark_current: float
        Dark current rate (e-/pix/s). Defaults to 0.00031.
    cic : float
        Clock induced charge (e-/pix/frame). Defaults to 0.016.
    read_noise : float
        Read noise (e-/pix/frame). Defaults to 110.
    bias : float
        Bias offset (e-). Defaults to 1500.  Because read noise is Gaussian with 
        a zero mean usually, negative variates are possible.  During read out,
        a voltage bias is added as a part of the conversion to digital numbers 
        (DN), which is 0 at minimum.  So to prevent clipping off negative values, 
        the bias should high enough (perhaps at least a few times the read noise).
    qe : float
        Quantum efficiency. Defaults to 0.9.
    cr_rate : float
        Cosmic ray rate (hits/cm^2/s). Defaults to 0.
    zff : float
        Free-field thickness of CCD (m). This is the perpendicular distance 
        that the cosmic ray travels before hitting the detector, which affects
        the size of the cosmic ray head.  Default is 8e-6m (for Roman CGI EMCCD).
    loc : float
        Location parameter for Landau distribution of total electrons delivered to sensor by cosmic ray. 
        Default is 1590e- (expected for Roman CGI EMCCD).  Together with scale 
        below gives a rough mean of 2360e- and most probable value (MPV) of 
        1360e-, which are the values expected for Roman CGI EMCCD at L2.  
    scale : float
        Scale parameter for Landau distribution of total electrons delivered to sensor by cosmic ray. 
        Default is 550e- (for Roman CGI EMCCD).
    oversample_factor : int
        Factor of oversampling of cosmic Gaussian over which to bin-sum to get 
        pixel values.  Default is 10.
    tail_length: int or str
        Desired tail length of cosmic ray (in pixels).  Defaults to 'roman', which gives a 
        tail length of about 40 for EM gain of 1000 (what is expected in flight 
        for the Roman Telescope) and shorter lengths for proportionally smaller EM gain.
        For example, a gain of 500 would mean a tail length of 20 in the 'roman' case.
    pixel_pitch : float
        Distance between pixel centers (m). Defaults to 13e-6.
    eperdn : float
        Electrons per dn. Defaults to 8.2.
    nbits : int
        Number of bits used by the ADC readout. Must be between 1 and 64,
        inclusive. Defaults to 14.
    numel_gain_register : int
        Number of gain register elements. For eventually modeling partial CIC.
        Defaults to 604.
    meta_path : str
        Full path of metadata.yaml.  If None, defaults to metadata.yaml in util
        folder.
    nonlin_path : str
        Path of nonlinearity correction file.  See doc string of
        nonlinearity.apply_relgains for details on the required
        format of the file.  If None, no application of
        nonlinearity is performed.  Defaults to None.
    flat_path : str
        Path of master flat file.  Assumed to be a FITS file for which the flat
        data resides in the first extension HDU.  The flat is assumed to be
        of image-area shape (specified by the metadata from meta_path),
        dark-subtracted, divided by k-gain, divided by EM gain, and desmeared. 
        If the input is None, no application of pixel nonuniformity is 
        performed.  Note that dead and/or poorly performing pixels can be 
        simulated with this input by using values much less than 1.  
        Defaults to None.  If the input master flat is compatible with the image 
        area, the data for the master flat is stored as self.flat_im.  If the 
        master flat is smaller (e.g., intended for sim_sub_frame()), 
        it is stored as self.flat.  
    hot_pixel_path : str
        Path of hot pixel file.  Assumed to be a FITS file for which the data 
        resides in the first extension HDU.  The values in the files should be
        factor of multiplication of the simulated dark_current applicable for 
        each pixel.  Regular pixels would have a value of 1, and hot/warm 
        pixels would have some value > 1 (though any value > 0 is allowed).   
        If the input is None, no hot pixels are simulated.  Defaults to None.
        If the input hot pixel map is compatible with the image 
        area, the data for the hot pixel map is stored as self.hot_pixel_map_im.  
        If the hot pixel map is smaller (e.g., intended for sim_sub_frame()), 
        it is stored as self.hot_pixel_map.  
    row_read_time : float
        Time in seconds for each row to move into the first register (same as 
        the time for each row to be clocked toward the register). This is used 
        to simulate smear on the image due to clocking during the exposure to 
        light.  Especially useful for shutterless EMCCDs.  If 0, no smear is 
        simulated.  Defaults to 223.5e-6 seconds (applicable to the Roman CGI).
    gain_CIC_Q : float or str
        Probability Q (or mean rate) of production of a clock-induced charge (CIC)
        in a given gain register stage. We call this "partial CIC".  
        Physically, Q < P, where P is the average probability of charge multiplication 
        for a single gain stage, and em_gain = (1+P)^numel_gain_register.  
        To simulate no partial CIC, let this input be 0.
        Defaults to 'roman', in which case P/40 is used, which is consistent with data 
        and gives negligible partial CIC for lower gains as expected.  
    gain_CIC_specs: dict or None
        This input supercedes gain_CIC_Q and renders the value of gain_CIC_Q 
        irrelevant.  This is used for specifying particular "hot" stages which source the 
        CIC produced in the gain register.  If None, gain_CIC_Q assumed for all
        gain register stages. If a dictionary is provided, the keys should be 
        integer-valued and be the number of stages until the end (e.g., 1 means 
        CIC appears in the last stage and gets clocked through that 1 gain stage),
        and the values for the dictionary should be the corresponding Q values. 
        Physically, Q < P, but we only know an average P from em_gain, so a 
        "hot" stage could have Q >= P, where em_gain = (1+P)^numel_gain_register.
        In fact, gain_CIC_specs can also be used to simulate charge traps in 
        the gain register (e.g., a small Q value in stage n could represent 
        charge capture, and a large Q value in stage n+2 could represent a 
        charge release on average 2 clockings later), though this should be 
        coordinated with the input tail_length, which also simulates this effect 
        but on a coarser scale.  Defaults to None.
    upstream_spill_prob : float or None
        For simulation of blooming (the overspill into neighboring rows from saturated pixels).  
        This parameter is the probability of charge in a saturated pixel to spill upstream (away from the readout direction) 
        to the next row during parallel clocking.  If None, no blooming is 
        simulated (perhaps there is near-perfect overspill protection in place for the detector).  
        If a float, it must be between 0 and 1.  If < 0.5, overspill to the row downstream is 
        more likely than upstream overspill.  Defaults to 0.7.  Upstream overspill 
        is more likely for the Roman CGI EMCCDs.
    fast_gain_mode : bool
        If True, a faster but less accurate method (uses Erlang/Gamma distribution for EM gain)
        of simulating the gain register is used.  If False, a slower but more accurate method 
        (marches each pixel through the gain register with binomial distribution) is used.
        The fast method is quite accurate for em_gain > 200.  Defaults to False.
    gain_stage_specs: dict or None
        This input is used for specifying particular gain stages which are "hot"
        with respect to the average probability of multiplication, P.  The input 
        em_gain specifies the average P value given no "hot" stages, and that value is applied to all 
        gain stages except for the ones specified by this gain_stage_specs.  
        If a dictionary is provided, the keys should be 
        integer-valued and be the number of stages until the end (e.g., 1 means 
        last stage), and the values for the dictionary should be the 
        corresponding probability values.  If the input fast_gain_mode
        is True, the average gain over all stages is computed and applied. 
        Defaults to None, in which case the same P value applies to all stages.
    fpn_path: str
        Inserting a FITS file that will serve as the fixed pattern noise (FPN) for the
        image.  Assumed to be a FITS file for which the FPN data is in units of electrons 
        (no voltage bias included) and resides in
        the primary HDU. If 'roman', the Roman
        CGI EXCAM FPN is used. If None, horizontal and
        vertical stripes of FPN are included, according to a normal distribution 
        specified by the bias_sigma_row and bias_sigma_col variables.  If one 
        of these is 0, no stripes will appear in that corresponding dimension (e.g., 
        no FPN pattern if fpn_path=None, bias_sigma_row=0, and bias_sigma_col=0).
    bias_sigma_row: float
        This number (in units of electrons) affects how large the normal distrubtion of FPN values for
        the rows of the FPN, with the input bias serving as the mean of the normal 
        distribution. This parameter is irrelevant if fpn_path is not None.
        The random seed for this variable depends on gain (em_gain), voltage bias (bias), exposure time (frametime), and dark current (dark_current).
        The FPN pattern is unique for a specific choice of these three parameters.
    bias_sigma_col: float
        This number (in units of electrons) affects how large the normal distrubtion of FPN values for
        the columns of the FPN, with the input bias serving as the mean of the normal 
        distribution. This parameter is irrelevant if fpn_path is not None.
        The random seed for this variable depends on gain (em_gain), voltage bias (bias), exposure time (frametime), and dark current (dark_current).
        The FPN pattern is unique for a specific choice of these three parameters.

    """
    def __init__(
        self,
        em_gain=1.,
        full_well_image=78000.,
        full_well_serial=105000.,
        dark_current=0.00031,
        cic=0.016,
        read_noise=110.,
        bias=1500.,
        qe=0.9,
        cr_rate=0.,
        zff=8e-6,
        loc=1590,
        scale=550, 
        oversample_factor=10,
        tail_length='roman',
        pixel_pitch=13e-6,
        eperdn=8.2,
        nbits=14,
        numel_gain_register=604,
        meta_path=None,
        nonlin_path=None,
        flat_path=None,
        hot_pixel_path=None, 
        row_read_time=223.5e-6,  # seconds
        gain_CIC_Q='roman',
        gain_CIC_specs=None,
        upstream_spill_prob=0.7,
        fast_gain_mode=False,
        gain_stage_specs=None,
        fpn_path= 'roman',
        bias_sigma_col=35,
        bias_sigma_row=35,
        **kwargs #accommodating other keyword args for backward compatibility
    ):     
        if row_read_time < 0:
            raise EMCCDDetectException('row_read_time must be >= 0 seconds.')
        if upstream_spill_prob is not None:
            if not (0 <= upstream_spill_prob <= 1):
                raise EMCCDDetectException('upstream_spill_prob must be between 0 and 1.')
        # specify same P value for all stages to initialize; this attribute present whether gain_stage_specs specified or not
        all_gain_stage_specs = {n:em_gain**(1/numel_gain_register) - 1 for n in range(1, numel_gain_register+1)}
        if gain_stage_specs is not None: 
            if not isinstance(gain_stage_specs, dict):
                raise EMCCDDetectException('gain_stage_specs must either be None or a dictionary.')
            if len(gain_stage_specs.values()) > numel_gain_register:
                raise EMCCDDetectException('The number of stages specified in gain_stage_specs is more than numel_gain_register.')
            if np.max(list(gain_stage_specs.keys())) > numel_gain_register:
                raise EMCCDDetectException('gain_stage_specs specifies a stage number beyond numel_gain_register.')
            if np.min(list(gain_stage_specs.keys())) < 1:
                raise EMCCDDetectException('gain_stage_specs specifies a stage number less than 1.')
            _, non_unique_counts = np.unique(gain_stage_specs.keys(), return_counts=True)
            if (non_unique_counts > 1).any():
                raise EMCCDDetectException('At least one gain stage was specified more than once in gain_stage_specs.')
            #overwrite with the specified values; and we do for loop in this way so that the dictionary is ordered, which is 
            # necessary in order for the branching process for gain application to be sequential
            for key, val in gain_stage_specs.items():
                if key//1 != key:
                    raise EMCCDDetectException('All keys in gain_stage_specs must be whole numbers.')
                if val < 0:
                    raise EMCCDDetectException("All values in gain_stage_specs must be non-negative.")
            for num_stages_left in range(1, numel_gain_register+1):
                if num_stages_left in gain_stage_specs.keys():
                    all_gain_stage_specs[num_stages_left] = gain_stage_specs[num_stages_left]
        self.avg_gain_P = np.sum(list(all_gain_stage_specs.values()))/numel_gain_register
        # reset what the value is based gain_CIC_specs
        em_gain = (1+self.avg_gain_P)**numel_gain_register
        if type(tail_length) != int and tail_length != 'roman':
            raise EMCCDDetectException("tail_length should be an integer or \'roman\'.")
        elif type(tail_length) == int:
            if tail_length < 0:
                raise EMCCDDetectException('tail_length cannot be negative.')             
        elif tail_length == 'roman':
            gain_avg = (1+self.avg_gain_P)**numel_gain_register
            tail_length = int(np.round(gain_avg*40/1000))
        if gain_CIC_Q != 'roman':
            if gain_CIC_Q > self.avg_gain_P:
                raise EMCCDDetectException('gain_CIC_Q >= P, where em_gain = '
                            '(1+P)^numel_gain_register. gain_CIC_Q must be < P.')
            self.gain_CIC_Q = gain_CIC_Q
        elif gain_CIC_Q == 'roman':
            self.gain_CIC_Q = self.avg_gain_P/40
        else:
            raise EMCCDDetectException('gain_CIC_Q must be float or \'roman\'.')
        if gain_CIC_specs is not None:
            if not isinstance(gain_CIC_specs, dict):
                raise EMCCDDetectException('gain_CIC_specs must either be None or a dictionary.')
            for key, val in gain_CIC_specs.items():
                if key//1 != key:
                    raise EMCCDDetectException('All keys in gain_CIC_specs must be whole numbers.')
                if val < 0:
                    raise EMCCDDetectException("All values in gain_stage_specs must be non-negative.")
            if len(gain_CIC_specs.values()) > numel_gain_register:
                raise EMCCDDetectException('The number of stages specified in gain_CIC_specs is more than numel_gain_register.')
            if np.max(list(gain_CIC_specs.keys())) > numel_gain_register:
                raise EMCCDDetectException('gain_CIC_specs specifies a stage number beyond numel_gain_register.')
            if np.min(list(gain_CIC_specs.keys())) < 1:
                raise EMCCDDetectException('gain_CIC_specs specifies a stage number less than 1.')
            _, non_unique_counts = np.unique(gain_CIC_specs.keys(), return_counts=True)
            if (non_unique_counts > 1).any():
                raise EMCCDDetectException('At least one gain stage was specified more than once in gain_CIC_specs.')
            # check that the average Q < P (i.e., the mean production rate of partial CIC electrons is less than the rate for gain multiplication)
            avg_CIC_Q = np.sum(list(gain_CIC_specs.values()))/numel_gain_register
            if avg_CIC_Q >= self.avg_gain_P: # won't cause problems for typical case where only a few stage values filled in
                raise EMCCDDetectException('The average Q over all gain stages must be < P, the average EM gain multiplication probability.')
        # If no metadata file path specified, default to metadata.yaml in util
        if meta_path is None:
            here = os.path.abspath(os.path.dirname(__file__))
            meta_path = Path(here, 'util', 'metadata.yaml')

        # Before inheriting base class, get metadata
        self.meta_path = meta_path
        self.meta = MetadataWrapper(self.meta_path)

        # instantiate remaining inputs not included in EMCCDDetectBase
        self.nonlin_path = nonlin_path
        self.row_read_time = row_read_time
        self.flat_path = flat_path
        self.hot_pixel_path = hot_pixel_path
        self.gain_CIC_specs = gain_CIC_specs
        self.tail_length = tail_length 
        self.zff = zff
        self.loc = loc
        self.scale = scale
        self.oversample_factor = oversample_factor
        self.upstream_spill_prob = upstream_spill_prob
        self.fast_gain_mode = fast_gain_mode
        self.gain_stage_specs = gain_stage_specs
        self.fpn_path = fpn_path
        self.bias_sigma_row = bias_sigma_row
        self.bias_sigma_col = bias_sigma_col

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
            eperdn=eperdn,
            nbits=nbits,
            numel_gain_register=numel_gain_register
        )
    
    def sim_full_frame(self, fluxmap, frametime):
        """Simulate a full detector frame.

        Note that the fluxmap provided must be the same size as the exposed
        detector pixels (specified in self.meta.geom['image']). A full frame
        including prescan and overscan regions will be made around the fluxmap.

        Parameters
        ----------
        fluxmap : array_like
            Input fluxmap, same shape as self.meta.geom['image'] (phot/pix/s).
        frametime : float
            Frame exposure time (s).

        Returns
        -------
        output_counts : array_like
            Detector output counts, including prescan/overscan (dn).

        """
        # Initialize the imaging area pixels
        imaging_area_zeros = self.meta.imaging_area_zeros.copy()
        # Embed the fluxmap within the imaging area. Create a mask for
        # referencing the input fluxmap subsection later
        fluxmap_full = self.meta.embed_im(imaging_area_zeros, 'image',
                                          fluxmap.copy())
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
        empty_element_m = (self.meta.mask('prescan')
                           + self.meta.mask('parallel_overscan')
                           + self.meta.mask('serial_overscan'))
        
        # Simulate serial clocking
        gain_counts = self.clock_serial(parallel_counts_full, empty_element_m)

        # Cap at full well capacity of gain register since any excess charge that isn't cleaned out via overscan (very unlikely) would presumably be placed in
        # the next frame, which is not simulated here. 
        gain_counts[gain_counts > self.full_well_serial] = self.full_well_serial

        # Simulate amplifier and adc redout
        output_dn = self.readout(gain_counts.reshape(parallel_counts_full.shape), frametime)

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

    def get_e_frame(self, frame_dn):
        """Take a raw frame output from EMCCDDetect and convert to a gain
        divided, bias subtracted frame in units of electrons.

        This will give the pre-readout image, i.e. the image in units of e- on
        the imaging plane.

        Parameters
        ----------
        frame_dn : array_like
            Raw output frame from EMCCDDetect, units of dn.

        Returns
        -------
        array_like
            Bias subtracted, gain divided frame in units of e-.

        """
        if hasattr(self, 'nonlin_path'):
            if self.nonlin_path is not None:
                nonlin_factors = apply_relgains(frame_dn, self.em_gain,
                                                self.nonlin_path)
                # correct for nonlin by dividing
                frame_dn = frame_dn/nonlin_factors
        return (frame_dn * self.eperdn - self.bias) / self.em_gain


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
    shot_noise_on=None
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
        Electron multiplying gain (e-/photoelectron).
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
        Apply shot noise. Defaults to None. [No longer supported as of v2.1.0.
        Input will have no effect.

    Returns
    -------
    serial_frame : array_like, float
        Detector output (e-).

    Notes
    -----
    The value for eperdn (electrons per dn) is hardcoded to 1. This is for
    legacy purposes, as the version 1.0.1 implementation output electrons
    instead of dn.

    The legacy version also has no gain register CIC, so
    numel_gain_register is irrelevant.

    The legacy version also had no ADC (it just output floats), so the number
    of bits is set as high as possible (64) and the output is converted to
    floats. This will still be different from the legacy version as there will
    no longer be negative numbers.

    """
    if shot_noise_on is not None:
        warnings.warn('Shot noise parameter no longer supported. Input has no '
                      'effect')

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
        eperdn=1.,
        nbits=64,
        numel_gain_register=604,
    )

    return emccd.sim_sub_frame(fluxmap, frametime).astype(float)
