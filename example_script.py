# -*- coding: utf-8 -*-
"""Cosmic removal simulation.

S Miller - UAH - 4-Feb-2020
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from emccd_detect.emccd_detect import emccd_detect
from emccd_detect.util.imagesc import imagesc


plt.close('all')

# Input frame
current_path = os.path.dirname(os.path.abspath(__file__))
fits_name = 'ref_frame.fits'
fluxmap = fits.getdata(os.path.join(current_path, 'emccd_detect', 'fits',
                       fits_name))

imagesc(fluxmap, 'Input Fluxmap')

# Simulation inputs
cr_rate = 0  # hits/cm^2/s (set to 0 for no cosmics; 5 for L2 expected)
frametime = 100.0  # seconds
em_gain = 1000.0  # setting the EM gain is by the user
bias = 0.0

qe = 0.9  # quantum efficiency
fwc_im = 50000.0  # full well capacity (image plane)
fwc_gr = 90000.0  # full well capacity (gain register)
dark_current = 0.005  # e-/pix/s
cic = 0.02  # e-/pix/frame
read_noise = 100  # e-/pix/frame -- amplifier noise (EMCCD CCD201 Type C)

gain_array = np.logspace(0, 4, 5, base=10)
for gain in gain_array:
    sim_im = emccd_detect(fluxmap, cr_rate, frametime, gain, bias, qe, fwc_im,
                          fwc_gr, dark_current, cic, read_noise)
    imagesc(sim_im, 'Gain: {:.0f}    RN: {:.0f}e-    t_fr: {:.0f}s'.format(
            gain, read_noise, frametime))

# Photon counting example
sim_im = emccd_detect(fluxmap, 0, frametime, gain, bias, qe, fwc_im,
                      fwc_gr, dark_current, cic, read_noise)

# For best performance, choose frametime such that average e/pix/frame in
# the region of interest (ROI) is roughly 0.1 e/pix/frame
