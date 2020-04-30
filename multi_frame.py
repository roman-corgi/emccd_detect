# -*- coding: utf-8 -*-
"""Generate multiple frames and write to fits files.

S Miller and B Nemati - UAH - 21-Feb-2020
"""
from __future__ import absolute_import, division, print_function

from os import path
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from emccd_detect.emccd_detect import emccd_detect
from emccd_detect.util.imagesc import imagesc

# Input fluxmap
fits_name = 'ref_frame.fits'
current_path = Path(path.dirname(__file__))
fits_path = Path(current_path, 'data', fits_name)
fluxmap = fits.getdata(fits_path)  # Input fluxmap (photons/pix/s)

# Number of images to create
nfiles = 3

# Plot output images
plot_images = True

# Make brights
path = '/Users/sammiller/Documents/GitHub/proc_cgi_frame/data/sim/brights/'
file_name = 'sim'
for i in range(nfiles):
    sim_im = emccd_detect(fluxmap,
                          frametime=100.,
                          em_gain=5000.,
                          full_well_image=20000.,
                          full_well_serial=48440.,
                          dark_current=0.0028,
                          cic=0.01,
                          read_noise=100.,
                          bias=5000.,
                          qe=0.9,
                          cr_rate=1.
                          )
    fits.writeto(Path(path, '{}{}.fits'.format(file_name, i)), sim_im)
    if plot_images:
        imagesc(sim_im, 'Output Image')

# Make darks
path = '/Users/sammiller/Documents/GitHub/proc_cgi_frame/data/sim/darks/'
file_name = 'sim'
for i in range(nfiles):
    sim_im = emccd_detect(np.zeros(fluxmap.shape),
                          frametime=100.,
                          em_gain=5000.,
                          full_well_image=20000.,
                          full_well_serial=48440.,
                          dark_current=0.0028,
                          cic=0.01,
                          read_noise=100.,
                          bias=5000.,
                          qe=0.9,
                          cr_rate=0.
                          )
    fits.writeto(Path(path, '{}{}.fits'.format(file_name, i)), sim_im)
    if plot_images:
        imagesc(sim_im, 'Output Image')

plt.show()
