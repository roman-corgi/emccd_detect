# -*- coding: utf-8 -*-
"""EMCCD Detector Simulation.

S Miller and B Nemati - UAH - 21-Feb-2020
"""
import os

from astropy.io import fits

from emccd_detect.emccd_detect import emccd_detect
from emccd_detect.util.imagesc import imagesc


# Input fluxmap
fits_name = 'ref_frame.fits'
current_path = os.path.dirname(os.path.abspath(__file__))
fits_path = os.path.join(current_path, 'fits', fits_name)
fluxmap = fits.getdata(fits_path)  # photons/pix/s

# Simulation inputs
exptime = 100.0  # Frame time (seconds)
gain = 1000.0  # CCD gain (e-/photon)
full_well_serial = 10000.0  # Serial register capacity (e-)
full_well = 60000.0  # Readout register capacity (e-)
dark_rate = 0.0056  # Dark rate (e-/pix/s)
cic_noise = 0.01  # Charge injection noise (e-/pix/frame)
read_noise = 100  # Read noise (e-/pix/frame)
bias = 0.0  # Bias offset (e-)
quantum_efficiency = 0.9
cr_rate = 5  # Cosmic ray rate (5 for L2) (hits/cm^2/s)
apply_smear = True  # Apply LOWFS readout smear

# Simulate single image
sim_im = emccd_detect(fluxmap, exptime, gain, full_well_serial, full_well,
                      dark_rate, cic_noise, read_noise, bias,
                      quantum_efficiency, cr_rate, apply_smear)

# Plot images
plot_images = True
if plot_images:
    imagesc(fluxmap, 'Input Fluxmap')
    imagesc(sim_im, 'Output Fluxmap\n'
            'Gain: {:.0f}    RN: {:.0f}e-    t_fr: {:.0f}s'.format(gain,
                                                                   read_noise,
                                                                   exptime))
