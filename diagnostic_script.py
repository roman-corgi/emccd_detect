# -*- coding: utf-8 -*-
"""Basic detector simulation.

B Nemati and S Miller - UAH - 18-Jan-2019
"""
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from emccd_detect.emccd_detect import emccd_detect
from emccd_detect.rand_em_gain import rand_em_gain
# from emccd_detect.util.auto_arrange_figures import auto_arrange_figures
from emccd_detect.util.histbn import histbn
from emccd_detect.util.imagesc import imagesc
from emccd_detect.util.peaks import peaks

plt.close('all')
# Set auto_arrange_figures inputs
figrows = 3
figcols = 4
display = 2

# input frame
roi_size = 200
max_flux = 100
temp = peaks(roi_size)**2  # dummy image inupt
fluxmap = max_flux * (temp - np.min(temp)) / (np.max(temp)-np.min(temp))

imagesc(fluxmap, 'input flux map, ph/s/pix')

# emccd_detect inputs
cr_rate = 5  # hits/cm^2/s (set to 0 for no cosmics; 5 for L2 expected)
frametime = 100.0  # seconds
em_gain = 100.0  # setting the EM gain is by the user
bias = 0.0

qe = 1.0  # quantum efficiency
fwc_im = 50000.0  # full well capacity (image plane)
fwc_gr = 90000.0  # full well capacity (gain register)
dark_current = 0.005  # e-/pix/s
cic = 0.02  # e-/pix/frame
read_noise = 120  # e-/pix/frame -- amplifier noise (EMCCD CCD201 Type C)

# Time the execution of a call
num_images = 1
sim1 = np.zeros((roi_size, roi_size, num_images))
t1 = time.time()
for image_n in range(0, num_images):
    sim1[:, :, image_n] = emccd_detect(fluxmap, 0.0, frametime, em_gain,
                                       bias, qe, fwc_im, fwc_gr, dark_current,
                                       cic, read_noise)
tt1 = time.time() - t1
print('Basic:    Simulating {:d} ({:d} x {:d}) images took {:.2f} seconds'
      '({:.2f} sec/image).\n'.format(image_n+1, roi_size, roi_size, tt1,
                                     tt1 / (image_n+1)))
imagesc(sim1[:, :, -1], 'sample simulated image with gain = ' + str(em_gain))


# Simulate random em gain + cosmic
sim2 = np.zeros((roi_size, roi_size, num_images))
t2 = time.time()
for image_n in range(0, num_images):
    sim2[:, :, image_n] = emccd_detect(fluxmap, cr_rate, frametime, em_gain,
                                       bias, qe, fwc_im, fwc_gr, dark_current,
                                       cic, read_noise)
tt2 = time.time() - t2
print('Combined: Simulating {:d} ({:d} x {:d}) images took {:.2f} seconds'
      '({:.2f} sec/image).\n'.format(image_n+1, roi_size, roi_size, tt2,
                                     tt2 / (image_n+1)))
imagesc(sim2[:, :, -1], 'cosmic rays on CCD201, MSL,'
        'rate = {:} CR/cm2/s'.format(cr_rate))

# Evaluate simulation
# The point of this is to see how the rand_em_gain distributions look
ntries = 2500
x = np.zeros(ntries)
nin_values = [1, 5, 10, 50]
for nin in nin_values:
    t3 = time.time()
    for it in range(0, ntries):
        x[it] = rand_em_gain(nin, em_gain)

    tt3 = time.time() - t3
    tper = tt3 / (ntries+1)
    h = histbn(x, 'all', 80)
    h.ax.grid(True)
    h.ax.set_title('Nin = {:} EMgain = {:} mean = {:.0f} ({:.3f} ms per'
                   'call)'.format(nin, em_gain, np.mean(x), tper*1000))

# Next look at images with increasing level of EM gain
ngains = 5
em_gains = np.logspace(0, np.log10(500), ngains)
for em_gain in em_gains:
    cr_rate = 0  # turn off for now
    this_image = emccd_detect(fluxmap, cr_rate, frametime, em_gain, bias, qe,
                              fwc_im, fwc_gr, dark_current, cic, read_noise)
    my_string = 'EMgain = {:.1f} (no CR)'.format(em_gain)
    imagesc(this_image, my_string)

plt.show()

# auto_arrange_figures(figrows, figcols, display)
