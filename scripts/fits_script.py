# -*- coding: utf-8 -*-
"""Basic detector simulation.

B Nemati and S Miller - UAH - 18-Jan-2019
"""
import os
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from emccd_detect.emccd_detect import emccd_detect
from emccd_detect.util.imagesc import imagesc
from emccd_detect.util.polar import cart2pol, rad_avg, rad_std


plt.close('all')

sampling = 0.423

# input frame
current_path = os.path.dirname(os.path.abspath(__file__))
fits_name = 'ref_frame.fits'
fluxmap = fits.getdata(os.path.join(current_path, 'emccd_detect', 'fits',
                       fits_name))
frame_size = fluxmap.shape[0]

rad_avgs, rho_vector_avg = rad_avg(fluxmap, 1/sampling)
rho_vector_avg = rho_vector_avg * sampling
rad_stds, rho_vector_std = rad_std(fluxmap, 1/sampling)
rho_vector_std = rho_vector_std * sampling

extent_edge = (frame_size/2)*sampling
extent = [-extent_edge, extent_edge, -extent_edge, extent_edge]

fm_fig, fm_ax = imagesc(fluxmap, 'input flux map, ph/s/pix', extent=extent)
fm_ax.set_xlabel('lam/D')
fm_ax.set_ylabel('lam/D')

# emccd_detect inputs
cr_rate = 0  # hits/cm^2/s (set to 0 for no cosmics; 5 for L2 expected)
frametime = 20.0  # seconds
em_gain = 5700.0  # setting the EM gain is by the user
bias = 0.0

qe = 0.845  # quantum efficiency
fwc_im = 50000.0  # full well capacity (image plane)
fwc_gr = 90000.0  # full well capacity (gain register)
dark_current = 0.00056  # e-/pix/s
cic = 0.02  # e-/pix/frame
read_noise = 120  # e-/pix/frame -- amplifier noise (EMCCD CCD201 Type C)

# Regular run (without using shot noise off flag)
num_images = 1
sim1 = np.zeros((frame_size, frame_size, num_images))
t1 = time.time()
for image_n in range(0, num_images):
    sim1[:, :, image_n] = emccd_detect(fluxmap, 0.0, frametime, em_gain,
                                       bias, qe, fwc_im, fwc_gr, dark_current,
                                       cic, read_noise)
tt1 = time.time() - t1
print('Simulating {:d} ({:d} x {:d}) images took {:.2f} seconds'
      '({:.2f} sec/image).\n'.format(image_n+1, frame_size, frame_size, tt1,
                                     tt1 / (image_n+1)))
sim1_sel = sim1[:, :, -1]
sim1_fig, sim1_ax = imagesc(sim1_sel, 'sample simulated image with gain = '
                            + str(em_gain), extent=extent)
sim1_ax.set_xlabel('lam/D')
sim1_ax.set_ylabel('lam/D')

# Shot noise off run
shot_noise_off = True
sim2 = np.zeros((frame_size, frame_size, num_images))
t1 = time.time()
for image_n in range(0, num_images):
    sim2[:, :, image_n] = emccd_detect(fluxmap, 0.0, frametime, em_gain,
                                       bias, qe, fwc_im, fwc_gr, dark_current,
                                       cic, read_noise, shot_noise_off)
tt1 = time.time() - t1
print('Simulating {:d} ({:d} x {:d}) images took {:.2f} seconds'
      '({:.2f} sec/image).\n'.format(image_n+1, frame_size, frame_size, tt1,
                                     tt1 / (image_n+1)))
sim2_sel = sim2[:, :, -1]
sim2_fig, sim2_ax = imagesc(sim2_sel, 'sample simulated image (shot noise off)'
                            ' with gain = ' + str(em_gain), extent=extent)
sim2_ax.set_xlabel('lam/D')
sim2_ax.set_ylabel('lam/D')

plt.figure()
plt.plot(rho_vector_avg, rad_avgs)
plt.xlabel('lam/D')
plt.ylabel('ph/s/pix')
plt.title('Fluxmap: Radial Average')

plt.figure()
plt.plot(rho_vector_std, rad_stds)
plt.xlabel('lam/D')
plt.ylabel('ph/s/pix')
plt.title('Fluxmap: Radial Standard Deviation')

plt.show()
