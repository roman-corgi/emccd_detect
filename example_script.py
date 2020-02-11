# -*- coding: utf-8 -*-
"""Gain and Photon Counting Simulation.

S Miller and B Nemati - UAH - 11-Feb-2020
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from emccd_detect.emccd_detect import emccd_detect
from emccd_detect.photon_count import photon_count
from emccd_detect.util.imagesc import imagesc


plt.close('all')

# Input fluxmap
current_path = os.path.dirname(os.path.abspath(__file__))
fits_name = 'ref_frame.fits'
fluxmap = fits.getdata(os.path.join(current_path, 'emccd_detect', 'fits',
                       fits_name))
imagesc(fluxmap, 'Input Fluxmap')


# Simulate frames with multiple different gains
# Simulation inputs
frametime = 100.0  # seconds
cr_rate = 0  # hits/cm^2/s (set to 0 for no cosmics; 5 for L2 expected)
bias = 0.0
qe = 0.9  # quantum efficiency
fwc_im = 50000.0  # full well capacity (image plane)
fwc_gr = 90000.0  # full well capacity (gain register)
dark_current = 0.005  # e-/pix/s
cic = 0.02  # e-/pix/frame
read_noise = 100  # e-/pix/frame -- amplifier noise (EMCCD CCD201 Type C)

len_array = 5
gain_array = np.logspace(0, 4, len_array, base=10)
for gain in gain_array:
    sim_im_gain = emccd_detect(fluxmap, cr_rate, frametime, gain, bias, qe,
                               fwc_im, fwc_gr, dark_current, cic, read_noise)
    imagesc(sim_im_gain,
            'Gain: {:.0f}    RN: {:.0f}e-    t_fr: {:.0f}s'.format(gain,
                                                                   read_noise,
                                                                   frametime))


# Simulate photon counting
gain = 1000.0
sim_im = emccd_detect(fluxmap, cr_rate, frametime, gain, bias, qe, fwc_im,
                      fwc_gr, dark_current, cic, read_noise)
imagesc(sim_im, 'Analogue')

pc_thresh = 100  # Photon counting threshold (to be multiplied by readnoise)
sim_in_pc = photon_count(sim_im, read_noise, pc_thresh)
imagesc(sim_in_pc, 'Photon Counted')


# Simulate co-added photon counted frames with short frametime
# For best performance, choose frametime such that average e/pix/frame in
# the region of interest (ROI) is roughly 0.1
frametime_short = 1.0
gain_short = 1000.0
pc_thresh_short = 10

n_frames = 100
sim_ims = np.zeros((fluxmap.shape[0], fluxmap.shape[1], n_frames))
sim_ims_pc = np.zeros(sim_ims.shape)
for i in range(n_frames):
    sim_ims[:, :, i] = emccd_detect(fluxmap, cr_rate, frametime_short,
                                    gain_short, bias, qe, fwc_im, fwc_gr,
                                    dark_current, cic, read_noise)
    sim_ims_pc[:, :, i] = photon_count(sim_ims[:, :, i], read_noise,
                                       pc_thresh_short)

co_added = np.sum(sim_ims, axis=2)
co_added_pc = np.sum(sim_ims_pc, axis=2)

imagesc(co_added, 'Co-Added Analogue')
imagesc(co_added_pc, 'Co-Added Photon Counted')
