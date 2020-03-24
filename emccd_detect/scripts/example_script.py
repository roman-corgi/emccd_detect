# -*- coding: utf-8 -*-
"""EMCCD Detector Simulation.

S Miller and B Nemati - UAH - 21-Feb-2020
"""
from __future__ import absolute_import, division, print_function

import sys
sys.path.append('.')
from os import path

from astropy.io import fits
import matplotlib.pyplot as plt

from emccd_detect.emccd_detect import emccd_detect
from emccd_detect.util.imagesc import imagesc

# Input fluxmap
fits_name = 'ref_frame.fits'
current_path = path.abspath(path.dirname(__file__))
fits_path = path.join(current_path, 'data', fits_name)
fluxmap = fits.getdata(fits_path)  # Input fluxmap (photons/pix/s)

# Simulation inputs
frametime = 100.  # Frame time (s)
em_gain = 100.  # CCD EM gain (e-/photon)
full_well_image = 60000.  # Image area full well capacity (e-)
full_well_serial = 10000.  # Serial (gain) register full well capacity (e-)
dark_current = 0.0056  # Dark current rate (e-/pix/s)
cic = 0.01  # Clock induced charge (e-/pix/frame)
read_noise = 100.  # Read noise (e-/pix/frame)
bias = 0.  # Bias offset (e-)
qe = 0.9  # Quantum efficiency
cr_rate = 5.  # Cosmic ray rate (5 for L2) (hits/cm^2/s)
pixel_pitch = 13e-6  # Distance between pixel centers (m)
shot_noise_on = True  # Apply shot noise

# Simulate single image
sim_im = emccd_detect(fluxmap, frametime, em_gain, full_well_image,
                      full_well_serial, dark_current, cic, read_noise, bias,
                      qe, cr_rate, pixel_pitch, shot_noise_on)

# Plot images
plot_images = True
if plot_images:
    imagesc(fluxmap, 'Input Fluxmap')

    subtitle = ('Gain: {:.0f}   Read Noise: {:.0f}e-   Frame Time: {:.0f}s'
                .format(em_gain, read_noise, frametime))
    imagesc(sim_im, 'Output Image\n' + subtitle)
    plt.show()