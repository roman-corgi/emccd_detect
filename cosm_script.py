# -*- coding: utf-8 -*-
"""Cosmic removal simulation.

S Miller - UAH - 4-Feb-2020
"""
import os

from astropy.io import fits

from emccd_detect.emccd_detect import emccd_detect
from emccd_detect.remove_cosmics import remove_cosmics
from emccd_detect.util.imagesc import imagesc


# Input frame
current_path = os.path.dirname(os.path.abspath(__file__))
fits_name = 'ref_frame.fits'
fluxmap = fits.getdata(os.path.join(current_path, 'emccd_detect', 'fits',
                       fits_name))
# Simulation inputs
cr_rate = 5  # hits/cm^2/s (set to 0 for no cosmics; 5 for L2 expected)
frametime = 100.0  # seconds
em_gain = 10.0  # setting the EM gain is by the user
bias = 0.0

qe = 1.0  # quantum efficiency
fwc_im = 50000.0  # full well capacity (image plane)
fwc_gr = 90000.0  # full well capacity (gain register)
dark_current = 0.005  # e-/pix/s
cic = 0.02  # e-/pix/frame
read_noise = 120  # e-/pix/frame -- amplifier noise (EMCCD CCD201 Type C)

# Simulate image with cosmic ray hits
sim_im = emccd_detect(fluxmap, cr_rate, frametime, em_gain, bias, qe, fwc_im,
                      fwc_gr, dark_current, cic, read_noise)

imagesc(sim_im)


sat_thresh = 0.99
plat_thresh = 0.95
cosm_filter = 3
tail_filter = 5

cleaned, cosm_mask, tail_mask = remove_cosmics(sim_im, fwc_gr, sat_thresh,
                                               plat_thresh, cosm_filter,
                                               tail_filter)

imagesc(cleaned)
