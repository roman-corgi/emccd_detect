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

studyCase = 'new'
# Simulation inputs
frameTime = 1.  # Frame time (seconds)
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

zeroFrame = np.zeros(fluxmap.shape)
npts = 55
pc_thresh = np.linspace(200, 1600, npts)
nthr = np.zeros(npts)
eps_thr = np.zeros(npts)
dark_an_mn = np.zeros(npts)
nobs_dk = np.zeros(npts)
rtrue_dk = np.zeros(npts)
bright_an_mn = np.zeros(npts)
nobs_br = np.zeros(npts)
rtrue_br = np.zeros(npts)
r_phe = np.zeros(npts)
for ithr in range(npts):
    # Threshold and photon count
    nthr[ithr] = pc_thresh[ithr] / read_noise

    eps_thr[ithr] = np.exp(-pc_thresh[ithr] / em_gain)

    # Dark frame
    darkFrame = emccd_detect(zeroFrame, frameTime, em_gain, full_well_image,
                             full_well_serial, dark_current, cic, read_noise,
                             bias, qe, cr_rate, pixel_pitch, True)
    dark_an_mn[ithr] = np.mean(darkFrame)
    # photon-count the dark frame
    dark_PC = zeroFrame
    dark_PC[darkFrame > pc_thresh[ithr]] = 1
    # the raw photon-counted frame needs to be corrected for inefficiencies
    # from thresholding and coincidence losses
    # observed mean rate after photon counting
    nobs_dk[ithr] = np.count_nonzero(dark_PC) / npix_across**2
    lambda_dk = -np.log(1 - (nobs_dk[ithr]/eps_thr[ithr]))
    rtrue_dk[ithr] = lambda_dk / frameTime

    # Bright frame
    brightFrame = emccd_detect(fluxmap, frameTime, em_gain, full_well_image,
                               full_well_serial, dark_current, cic, read_noise,
                               bias, qe, cr_rate, pixel_pitch, True)
    bright_an_mn[ithr] = np.mean(brightFrame)
    bright_PC = zeroFrame
    bright_PC[brightFrame > pc_thresh[ithr]] = 1
    # the raw photon-counted frame needs to be corrected for inefficiencies
    # from thresholding and coincidence losses
    # observed mean rate after photon counting
    nobs_br[ithr] = np.count_nonzero(bright_PC) / npix_across**2
    lambda_br = -np.log(1 - (nobs_br[ithr]/eps_thr[ithr]))
    rtrue_br[ithr] = lambda_br / frameTime

    # photo-electron rate
    r_phe[ithr] = rtrue_br[ithr] - rtrue_dk[ithr]

    if ithr == 1:
        imagesc(bright_PC, cmap='gray')
        imagesc(brightFrame)

# Threshold efficincy for n=1 and n=2 EM probablity distributions
def eps_th1(x, g): return np.exp(-x/g)
def eps_th2(x, g): return (1+x/g) * np.exp(-x/g)
def eps_th3(x, g): return (1+(x/g)+0.5*(x/g)**2) * np.exp(-x/g)
def pdfEM(x, g, n): return x**(n-1) * np.exp(-x/g) / (g**n*np.factorial(n-1))

pp1 = poisson.pmf(r_phe, 1)
pp2 = poisson.pmf(r_phe, 2)
pp3 = poisson.pmf(r_phe, 3)
eth1 = eps_th1(nthr*read_noise, em_gain)
eth2 = eps_th2(nthr*read_noise, em_gain)
eth3 = eps_th3(nthr*read_noise, em_gain)
overcountEst2 = (pp1*eth1 + pp2*eth2) / ((pp1+pp2) * eth1)
overcountEst3 = (pp1*eth1 + pp2*eth2 + pp3*eth3) / ((pp1+pp2+pp3)*eth1)


plt.figure()
plt.plot(nthr, nobs_br/frameTime, nthr, r_phe, nthr, flux*np.ones(1, npts))
plt.grid(True)
plt.legend('Observed', 'Corrected', 'Actual')
plt.xlabel('threshold factor')
plt.ylabel('rates, e/pix/s')
plt.title('RN=' + str(read_noise) + ' emG=' + str(em_gain) + ' FWCs=' + str(full_well_serial/1000) + 'k')

plt.figure()
plt.plot(nthr, eps_thr)
plt.grid(True)
plt.xlabel('threshold factor')
plt.ylabel('threshold effeciency')
plt.title('Assuming all pixels are 1 or 0 real ph-e''s')

plt.figure()
plt.plot(nthr, overcountEst2)
plt.grid(True)
plt.xlabel('threshold factor')
plt.ylabel('PC over-count factor')

plt.figure()
plt.plot(nthr, nobs_br/frameTime, '.-', nthr, r_phe, '.-', nthr, flux*np.ones(1, npts),
         nthr, r_phe/overcountEst2, '.-', nthr, r_phe/overcountEst3, '.-')
plt.grid(True)
plt.legend('Raw Phot Cnt', 'thr, CL corr', 'Actual', '+ovrcnt corr', '+n3 corr')
plt.xlabel('threshold factor')
plt.ylabel('rates, e/pix/s')
plt.title('RN=' + str(read_noise) + ' emG=' + str(em_gain) + ' FWCs=' + str(full_well_serial/1000) + 'k')

actualc = flux*np.ones(1, npts)

plt.figure()
plt.plot(nthr, r_phe/actualc, '.-', nthr, r_phe/overcountEst2/actualc, '.-',
         nthr, r_phe/overcountEst3/actualc, '.-', nthr, np.ones(1, npts))
plt.grid(True)
plt.legend('thr, CL corr', '+ovrcnt corr', '+n3 corr')
plt.xlabel('threshold factor')
plt.ylabel('rate/actual')
plt.title('RN=' + str(read_noise) + ' emG=' + str(em_gain) + ' FWCs=' + str(full_well_serial/1000) + 'k')
