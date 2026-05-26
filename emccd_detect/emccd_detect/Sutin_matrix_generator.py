# -*- coding: utf-8 -*-
"""Generate random numbers according to EM gain pdfs."""

import numpy as np
from scipy import special
from scipy.special import gammaln
#from emccd_detect.partial_CIC_MLE import _LogPn, _LogGamma
try:
    from joblib import Parallel, delayed
    import multiprocessing
except:
    pass
from partial_CIC_MLE import _LogPn, _LogGamma
from astropy.io import fits
import os


def Brian_Sutin(gain, M, nmax, w, x):
    '''Using Brian Sutin's formulation of Matsuo, 
    a form much easier to compute.
    '''
    num_proc = multiprocessing.cpu_count() #XXX add class attribute for num_proc? 
    P0 = np.zeros(nmax+1)
    P0[w] = 1 # w is the number of electrons incident on the gain register
    B = np.zeros((nmax+1, nmax+1))
    Q = gain**(1/M) -1 #Q is the probability of multiplication in a given stage
    if Q >= 1:
        raise ValueError('Gain and M values are not compatible. Q must be <1.')
    def compute_B_row(n):
        for k in range(int(np.ceil(n/2)), n+1):
            #if k <= n and n-k<= k:
            B[k,n] = np.exp(gammaln(k+1) - gammaln(n-k+1) - gammaln(2*k-n+1) + (2*k-n)* np.log(1-Q) + (n-k)*np.log(Q))
    
    Parallel(n_jobs=num_proc, prefer='threads')(delayed(compute_B_row)(n) for n in range(1, nmax+1))


    # for n in range(1, nmax+1):
    #     for k in range(int(np.ceil(n/2)), n+1):
    #         #if k <= n and n-k<= k:
    #         B[k,n] = np.exp(gammaln(k+1) - gammaln(n-k+1) - gammaln(2*k-n+1) + (2*k-n)* np.log(1-Q) + (n-k)*np.log(Q))
    
    Bpower = np.linalg.matrix_power(B, M)
    P = P0.dot(Bpower)
    return P[x]

if __name__ == '__main__':


    gain_range = np.linspace(2, 7000, num=10)
    M_range = np.linspace(30, 900, num=10).astype(int)
    w = np.linspace(1, 100, num=10).astype(int)
    dimension = int(gain_range.max() * w.max() * 1.2)
    gain_arrays = []
    for gain in gain_range:
        M_arrays = []
        for M in M_range:
            w_arrays = []
            for w_i in w:
                printout = 'gain: ' + str(gain) + ', M: ' + str(M) + ', w:' + str(w_i)
                print('Calculating ' + printout)
                nmax=int(gain*w_i*1.2)
                x_arr = np.arange(0, nmax)
                prob = Brian_Sutin(gain=gain, M=M, nmax=nmax, w=w_i, x=x_arr)
                output = np.zeros(dimension)
                output[0:len(prob)] = prob
                w_arrays.append(output)
            M_arrays.append(w_arrays)
        gain_arrays.append(M_arrays)

    hdu1 = fits.PrimaryHDU(data=np.stack(gain_arrays))
    hdu2 = fits.ImageHDU(data=gain_range, name='GAIN_VALS')
    hdu3 = fits.ImageHDU(data=M_range, name='M_VALS')
    hdu4 = fits.ImageHDU(data=w, name='W_VALS')
    
    hdul = fits.HDUList([hdu1, hdu2, hdu3, hdu4])
    hdul.writeto('sutin_matrix_output.fits', overwrite=True)
    

    pass