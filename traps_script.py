# -*- coding: utf-8 -*-
"""Script to show the effect of traps on detector output.

S Miller and B Nemati - UAH - 21-Feb-2020
"""
from __future__ import absolute_import, division, print_function

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from emccd_detect.emccd_detect_old import emccd_detect_old
from emccd_detect.emccd_detect import EMCCDDetect, emccd_detect
from emccd_detect.util.imagesc import imagesc
from arcticpy.roe import ROE
from arcticpy.ccd import CCD
from arcticpy.traps import Trap
from arcticpy.main import model_for_HST_ACS


here = os.path.abspath(os.path.dirname(__file__))

# Input fluxmap
fits_name = 'sci_frame.fits'
fits_path = Path(here, 'data', fits_name)
fluxmap = fits.getdata(fits_path).astype(float)  # Input fluxmap (photons/pix/s)

# Put fluxmap in 1024x1024 image section
image = np.zeros((1024, 1024)).astype(float)
image[0:fluxmap.shape[0], 0:fluxmap.shape[1]] = fluxmap

# Simulation inputs
frametime = 10.  # Frame time (s)
em_gain = 5000.  # CCD EM gain (e-/photon)
full_well_image = 50000.  # Image area full well capacity (e-)
full_well_serial = 90000.  # Serial (gain) register full well capacity (e-)
dark_current = 0.0028  # Dark current rate (e-/pix/s)
cic = 0.01  # Clock induced charge (e-/pix/frame)
read_noise = 100.  # Read noise (e-/pix/frame)
bias = 0.  # Bias offset (e-)
qe = 0.9  # Quantum efficiency
cr_rate = 0.  # Cosmic ray rate (5 for L2) (hits/cm^2/s)
pixel_pitch = 13e-6  # Distance between pixel centers (m)
shot_noise_on = True  # Apply shot noise

# Hubble launch date
date = 2452334.5
traps, ccd, roe = model_for_HST_ACS(date)

# Use class
emccd = EMCCDDetect(
    meta_path=Path(here, 'data', 'metadata.yaml'),
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
    shot_noise_on=shot_noise_on
)

# No traps
sim_frame_notrap = emccd.sim_sub_frame(fluxmap, frametime)

expected_rate = emccd.mean_expected_rate

# Traps
emccd.update_cti(ccd=ccd, roe=roe, traps=traps, express=1)
sim_frame_trap = emccd.sim_sub_frame(fluxmap, frametime)

# nframes = 1000
# sim_frame_notraps = []
# sim_frame_traps = []
# path = r'C:\Users\smiller\CAO1 Dropbox\CAO_Team1\X_Data\Traps'
# for i in range(nframes):
#     sim_frame_notraps = emccd.sim_sub_frame(fluxmap, frametime=10)

#     fits.writeto(Path(path, f'sim_frame_notraps_{i}.fits'), sim_frame_notraps.astype(np.int32),
#                  overwrite=True)


# emccd.update_cti(ccd=ccd, roe=roe, traps=traps, express=1)
# for i in range(nframes):
#     sim_frame_traps = emccd.sim_sub_frame(fluxmap, frametime=30)

#     fits.writeto(Path(path, f'sim_frame_traps_{i}.fits'), sim_frame_traps.astype(np.int32),
#                  overwrite=True)

# write_to_file = False
# if write_to_file:
#     # path = '/Users/sammiller/Documents/GitHub/proc_cgi_frame/data/sim/'
#     path = '.'
#     fits.writeto(Path(path, 'sim.fits'), sim_frame.astype(np.int32),
#                  overwrite=True)

# Plot images
plot_images = True
if plot_images:
    imagesc(fluxmap, 'Input Fluxmap (phot/pix/s)')

    notrap = sim_frame_notrap/emccd.em_gain
    trap = sim_frame_trap/emccd.em_gain

    vmin = np.min(expected_rate)
    vmax = np.max(expected_rate)

    subtitle = f'Frametime = {int(frametime)}s'

    imagesc(expected_rate, 'Mean Expected Rate (e-/pix)\n' + subtitle)
    imagesc(notrap, 'Output Without Traps (e-/pix)\n' + subtitle,
            vmin=vmin, vmax=vmax)
    imagesc(trap, 'Output With Traps (e-/pix)\n' + subtitle,
            vmin=vmin, vmax=vmax)

    # for im in ims:
    #     subtitle = ('Gain: {:.0f}   Read Noise: {:.0f}e-   Frame Time: {:.0f}s'
    #                 .format(em_gain, read_noise, frametime))
    #     imagesc(im, 'Output Image\n' + subtitle)

    plt.show()
