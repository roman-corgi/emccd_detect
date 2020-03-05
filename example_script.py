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
fluxmap = fits.getdata(fits_path)

# Simulation inputs
frametime = 100.0  # seconds
gain = 1000.0
cr_rate = 10  # hits/cm^2/s (set to 0 for no cosmics; 5 for L2 expected)
bias = 0.0
qe = 0.9  # quantum efficiency
fwc_im = 50000.0  # full well capacity (image plane)
fwc_gr = 90000.0  # full well capacity (gain register)
dark_current = 0.005  # e-/pix/s
cic = 0.02  # e-/pix/frame
read_noise = 100  # e-/pix/frame -- amplifier noise (EMCCD CCD201 Type C)

# Simulate single image
sim_im = emccd_detect(fluxmap, cr_rate, frametime, gain, bias, qe, fwc_im,
                      fwc_gr, dark_current, cic, read_noise)

# Plot images
plot_images = True
if plot_images:
    imagesc(fluxmap, 'Input Fluxmap')
    imagesc(sim_im, 'Output Fluxmap\n'
            'Gain: {:.0f}    RN: {:.0f}e-    t_fr: {:.0f}s'.format(gain,
                                                                   read_noise,
                                                                   frametime))
