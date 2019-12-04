# -*- coding: utf-8 -*-
"""Basic detector simulation.

B Nemati and S Miller - UAH - 18-Jan-2019
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from emccd_detect.emccd_detect import emccd_detect


plt.close('all')

sampling = 0.423

# input frame
frame_size = 65
fluxmap_i = np.ones([frame_size, frame_size])

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

ENF = np.sqrt(2)

# step = 0.1
# flat_rate = np.arange(0, 10) * step
flat_rate = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
sim_error = np.zeros(len(flat_rate))
sim_error_nofwc = np.zeros(len(flat_rate))
calc_error = np.zeros(len(flat_rate))
n = 0
for rate in flat_rate:
    fluxmap = fluxmap_i * rate
    sim = emccd_detect(fluxmap, 0.0, frametime, em_gain, bias, qe, fwc_im,
                       fwc_gr, dark_current, cic, read_noise)
    sim = sim / em_gain
    sim_error[n] = np.std(sim)

    sim_nofwc = emccd_detect(fluxmap, 0.0, frametime, em_gain, bias, qe,
                             np.inf, np.inf, dark_current, cic, read_noise)
    sim_nofwc = sim_nofwc / em_gain
    sim_error_nofwc[n] = np.std(sim_nofwc)

    calc_error[n] = np.sqrt(ENF**2 * (rate*frametime*qe + (dark_current*frametime + cic)) + (read_noise/em_gain)**2)
    n += 1

plt.plot(flat_rate, sim_error, '--', linewidth=1, marker='o')
plt.plot(flat_rate, sim_error_nofwc, '--', linewidth=1, marker='s')
plt.plot(flat_rate, calc_error, linewidth=1, marker='o', fillstyle='none')

plt.xlabel('Flat Frame Input Rate (e-/pix/s)')
plt.ylabel('Pixel Noise, Standard Deviation (e-)')
plt.title('Detector Simulator Error vs. Expectation')

plt.legend(['Detector Code', 'Detector Code (no FWC)', 'Analytical (EB)'])

plt.show()
