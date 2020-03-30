"""EMCCD Detector Simulation.

S Miller and B Nemati - UAH - 21-Feb-2020
"""
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

from emccd_detect.emccd_detect import emccd_detect
from emccd_detect.util.imagesc import imagesc

# Input fluxmap
npix_across = 20
flux = 0.07  # photns/pix/s
fluxmap = flux * np.ones([npix_across, npix_across])

# Simulation inputs
frametime = 1.  # Frame time (s)
em_gain = 6000.  # CCD EM gain (e-/photon)
full_well_image = 60000.  # Image area full well capacity (e-)
full_well_serial = 90000.  # Serial (gain) register full well capacity (e-)
dark_current = 0.00028  # Dark  current rate (e-/pix/s)
cic = 0.02  # Clock induced charge (e-/pix/frame)
read_noise = 100.  # Read noise (e-/pix/frame)
bias = 0.  # Bias offset (e-)
qe = 1.  # Quantum effiency
cr_rate = 0.  # Cosmic ray rate (5 for L2) (hits/cm^2/s)
pixel_pitch = 13e-6  # Distance between pixel centers (m)

# Threshold and photon count
zero_frame = np.zeros(fluxmap.shape)
npts = 55
pc_thresh = np.linspace(200, 1600, npts)
eps_thr = np.zeros(npts)
nobs_dk = np.zeros(npts)
nobs_br = np.zeros(npts)
r_phe = np.zeros(npts)
for i in range(npts):
    # Threshold efficiency
    eps_thr[i] = np.exp(-pc_thresh[i] / em_gain)

    # Dark frame
    dark_frame = emccd_detect(zero_frame, frametime, em_gain, full_well_image,
                              full_well_serial, dark_current, cic, read_noise,
                              bias, qe, cr_rate, pixel_pitch, True)
    # Photon-count
    dark_pc = zero_frame
    dark_pc[dark_frame > pc_thresh[i]] = 1
    # Correct for inefficiencies from thresholding and coincidence loss
    nobs_dk[i] = np.count_nonzero(dark_pc) / npix_across**2
    lambda_dk = -np.log(1 - (nobs_dk[i]/eps_thr[i]))
    rtrue_dk = lambda_dk / frametime  # Observed mean rate after photon counting

    # Bright frame
    bright_frame = emccd_detect(fluxmap, frametime, em_gain, full_well_image,
                                full_well_serial, dark_current, cic, read_noise,
                                bias, qe, cr_rate, pixel_pitch, True)
    # Photon-count
    bright_pc = zero_frame
    bright_pc[bright_frame > pc_thresh[i]] = 1
    # Correct for inefficiencies from thresholding and coincidence loss
    nobs_br[i] = np.count_nonzero(bright_pc) / npix_across**2
    lambda_br = -np.log(1 - (nobs_br[i]/eps_thr[i]))
    rtrue_br = lambda_br / frametime  # Observed mean rate after photon counting

    # Photo-electron rate
    r_phe[i] = rtrue_br - rtrue_dk

# Threshold efficincy for n=1 and n=2 EM probablity distributions
sigma_thr = pc_thresh / read_noise
def eps_th1(x, g): return np.exp(-x/g)
def eps_th2(x, g): return (1+x/g) * np.exp(-x/g)
def eps_th3(x, g): return (1+(x/g)+0.5*(x/g)**2) * np.exp(-x/g)
def pdfEM(x, g, n): return x**(n-1) * np.exp(-x/g) / (g**n*np.factorial(n-1))

pp1 = poisson.pmf(r_phe, 1)
pp2 = poisson.pmf(r_phe, 2)
pp3 = poisson.pmf(r_phe, 3)
eth1 = eps_th1(sigma_thr*read_noise, em_gain)
eth2 = eps_th2(sigma_thr*read_noise, em_gain)
eth3 = eps_th3(sigma_thr*read_noise, em_gain)
overcount_est2 = (pp1*eth1 + pp2*eth2) / ((pp1+pp2) * eth1)
overcount_est3 = (pp1*eth1 + pp2*eth2 + pp3*eth3) / ((pp1+pp2+pp3)*eth1)


plt.figure()
plt.plot(sigma_thr, nobs_br/frametime,
         sigma_thr, r_phe,
         sigma_thr, flux*np.ones(1, npts))
plt.grid(True)
plt.legend('Observed', 'Corrected', 'Actual')
plt.xlabel('threshold factor')
plt.ylabel('rates, e/pix/s')
plt.title('RN={:d} emG={:d} FWCs={:d}k'.format(int(read_noise), int(em_gain),
                                               int(full_well_serial/1000)))

plt.figure()
plt.plot(sigma_thr, eps_thr)
plt.grid(True)
plt.xlabel('threshold factor')
plt.ylabel('threshold effeciency')
plt.title('Assuming all pixels are 1 or 0 real ph-e''s')

plt.figure()
plt.plot(sigma_thr, overcount_est2)
plt.grid(True)
plt.xlabel('threshold factor')
plt.ylabel('PC over-count factor')

plt.figure()
plt.plot(sigma_thr, nobs_br/frametime, '.-',
         sigma_thr, r_phe, '.-',
         sigma_thr, flux*np.ones(1, npts),
         sigma_thr, r_phe/overcount_est2, '.-',
         sigma_thr, r_phe/overcount_est3, '.-')
plt.grid(True)
plt.legend('Raw Phot Cnt', 'thr, CL corr', 'Actual', '+ovrcnt corr', '+n3 corr')
plt.xlabel('threshold factor')
plt.ylabel('rates, e/pix/s')
plt.title('RN={:d} emG={:d} FWCs={:d}k'.format(int(read_noise), int(em_gain),
                                               int(full_well_serial/1000)))

actualc = flux*np.ones(1, npts)
plt.figure()
plt.plot(sigma_thr, r_phe/actualc, '.-',
         sigma_thr, r_phe/overcount_est2/actualc, '.-',
         sigma_thr, r_phe/overcount_est3/actualc, '.-',
         sigma_thr, np.ones(1, npts))
plt.grid(True)
plt.legend('thr, CL corr', '+ovrcnt corr', '+n3 corr')
plt.xlabel('threshold factor')
plt.ylabel('rate/actual')
plt.title('RN={:d} emG={:d} FWCs={:d}k'.format(int(read_noise), int(em_gain),
                                               int(full_well_serial/1000)))
