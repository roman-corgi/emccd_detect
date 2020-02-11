# -*- coding: utf-8 -*-
"""Test emccd_detect runtime."""
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from emccd_detect.emccd_detect import emccd_detect
# from emccd_detect.util.auto_arrange_figures import auto_arrange_figures
from emccd_detect.util.histbn import histbn
from emccd_detect.util.peaks import peaks

plt.close('all')

# input frame
roi_size = 200
max_flux = 100
temp = peaks(roi_size)**2  # dummy image inupt
fluxmap = max_flux * (temp - np.min(temp)) / (np.max(temp)-np.min(temp))

# emccd_detect inputs
cr_rate = 5.0  # hits/cm^2/s (set to 0 for no cosmics; 5 for L2 expected)
frametime = 100.0  # seconds

bias = 0.0

qe = 1.0  # quantum efficiency
fwc_im = 50000.0  # full well capacity (image plane)
fwc_gr = 90000.0  # full well capacity (gain register)
dark_current = 0.005  # e-/pix/s
cic = 0.02  # e-/pix/frame
read_noise = 120  # e-/pix/frame -- amplifier noise (EMCCD CCD201 Type C)

num_images = 100
em_gains = [50, 100, 1000]
h = np.zeros(len(em_gains))
i = 0
for gain in em_gains:
    times = np.zeros(num_images)
    for image_n in range(0, num_images):
        t = time.time()
        sim = emccd_detect(fluxmap, cr_rate, frametime, gain, bias, qe,
                           fwc_im, fwc_gr, dark_current, cic, read_noise)
        tt = time.time() - t
        times[image_n] = tt
    histbn(times)

    i += 1

plt.show()

# auto_arrange_figures()
