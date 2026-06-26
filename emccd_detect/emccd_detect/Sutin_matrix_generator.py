# -*- coding: utf-8 -*-
"""Generate random numbers according to EM gain pdfs."""

import numpy as np
from scipy import special, sparse
from scipy.special import gammaln
from scipy.interpolate import RegularGridInterpolator
#from emccd_detect.partial_CIC_MLE import _LogPn, _LogGamma
try:
    from joblib import Parallel, delayed
    import multiprocessing
except:
    pass
from partial_CIC_MLE import _LogPn, _LogGamma
from astropy.io import fits
from rand_em_gain import exact_gain_PDF
import os



def Brian_Sutin(gain, M, w, x):
    '''Using Brian Sutin's formulation of Matsuo, 
    a form much easier to compute.
    '''
    nmax = x.max()
    P0 = np.zeros(nmax+1)
    P0[w] = 1 # w is the number of electrons incident on the gain register
    B = np.zeros((nmax+1, nmax+1))
    Q = gain**(1/M) -1 #Q is the probability of multiplication in a given stage
    if Q >= 1:
        raise ValueError('Gain and M values are not compatible. Q must be <1.')
    n, k = np.meshgrid(np.arange(nmax+1), np.arange(nmax+1))
    B[k,n] = np.exp(gammaln(k+1) - gammaln(n-k+1) - gammaln(2*k-n+1) + (2*k-n)* np.log(1-Q) + (n-k)*np.log(Q))
    inds=np.where(k>n)
    B[inds] = 0 

    Bpower = np.linalg.matrix_power(B, M)
    P = P0.dot(Bpower)
    return P

def Brian_Sutin_orig(gain, M, nmax, w, x, num_proc=None):
    '''Using Brian Sutin's formulation of Matsuo, 
    a form much easier to compute.
    '''
    if num_proc is None:
        num_proc = multiprocessing.cpu_count() 
    P0 = np.zeros(nmax+1)
    P0[w] = 1 # w is the number of electrons incident on the gain register
    B = np.zeros((nmax+1, nmax+1))
    Q = gain**(1/M) -1 #Q is the probability of multiplication in a given stage
    if Q >= 1:
        raise ValueError('Gain and M values are not compatible. Q must be <1.')
    def compute_B_row(n):
        for k in range(nmax+1): #range(int(np.ceil(n/2)), n+1):
            if k <= n and n-k<= k:
                B[k,n] = np.exp(gammaln(k+1) - gammaln(n-k+1) - gammaln(2*k-n+1) + (2*k-n)* np.log(1-Q) + (n-k)*np.log(Q))
    
    Parallel(n_jobs=num_proc, prefer='threads')(delayed(compute_B_row)(n) for n in range(1, nmax+1))


    # for n in range(1, nmax+1):
    #     for k in range(int(np.ceil(n/2)), n+1):
    #         #if k <= n and n-k<= k:
    #         B[k,n] = np.exp(gammaln(k+1) - gammaln(n-k+1) - gammaln(2*k-n+1) + (2*k-n)* np.log(1-Q) + (n-k)*np.log(Q))
    
    Bpower = np.linalg.matrix_power(B, M)
    P = P0.dot(Bpower)
    return P[x]


def interpolate_Brian_Sutin(gain, M, w, x, gain_arrays, gain_range, M_range, w_range):
    """Interpolate precomputed Brian_Sutin probability arrays along gain, M, and w axes.

    gain_arrays: array-like with shape (len(gain_range), len(M_range), len(w_range), dimension)
    gain_range, M_range, w_range: 1D arrays of parameter grid used to build gain_arrays
    Returns probabilities for indices x (array-like) for requested gain, M, w.
    """
    gain_arr = np.asarray(gain_arrays)
    gain_range = np.asarray(gain_range)
    M_range = np.asarray(M_range)
    w_range = np.asarray(w_range)

    # ensure parameters within bounds
    if gain < gain_range.min() or gain > gain_range.max():
        raise ValueError('gain outside provided gain_range')
    if M < M_range.min() or M > M_range.max():
        raise ValueError('M outside provided M_range')
    if w < w_range.min() or w > w_range.max():
        raise ValueError('w outside provided w_range')

    x = np.asarray(x)
    # ensure integer indices for x
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError('x must be integer indices')

    max_len = gain_arr.shape[3]
    if x.max() >= max_len:
        raise IndexError('Requested x exceeds stored probability length')

    # interpolate for each x index across gain, M, and w
    probs = np.empty(x.shape, dtype=float)

    for i, xi in enumerate(x):
        # extract 3D slice for this x index: shape (len(gain_range), len(M_range), len(w_range))
        data_3d = gain_arr[:, :, :, int(xi)]
        
        # interpolate along w_range first
        data_2d = np.zeros((len(gain_range), len(M_range)))
        for gi in range(len(gain_range)):
            for mi in range(len(M_range)):
                data_2d[gi, mi] = np.interp(w, w_range, data_3d[gi, mi, :])
        
        # interpolate along M_range
        data_1d = np.zeros(len(gain_range))
        for gi in range(len(gain_range)):
            data_1d[gi] = np.interp(M, M_range, data_2d[gi, :])
        
        # interpolate along gain_range
        probs[i] = np.interp(gain, gain_range, data_1d)

    return probs


def interpolate_Brian_Sutin_multi(gain, M, w, x, gain_arrays, gain_range, M_range, w_range, method='pchip'):
    """Interpolate precomputed Brian_Sutin probability arrays using trilinear
    interpolation across gain, M, and w axes for all x indices simultaneously.
    gain_arrays shape: (len(gain_range), len(M_range), len(w_range), dimension)
    Returns probabilities for indices x (array-like) for requested gain, M, w.
    """
    gain_arr = np.asarray(gain_arrays)
    gain_arr[np.isnan(gain_arr)] = 0
    gain_range = np.asarray(gain_range)
    M_range = np.asarray(M_range)
    w_range = np.asarray(w_range)

    # bounds checks
    if gain < gain_range.min() or gain > gain_range.max():
        raise ValueError('gain outside provided gain_range')
    if M < M_range.min() or M > M_range.max():
        raise ValueError('M outside provided M_range')
    if w < w_range.min() or w > w_range.max():
        raise ValueError('w outside provided w_range')

    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError('x must be integer indices')

    max_len = gain_arr.shape[3]
    if x.max() >= max_len:
        raise IndexError('Requested x exceeds stored probability length')

    # Prepare grid interpolator for each x slice simultaneously by treating
    # the first three axes as the regular grid and the last axis as separate
    # data channels. We'll reshape to (npoints, ) for each x and evaluate at
    # the single query point (gain, M, w).
    # Create interpolator that returns vector values using bounds_error=True
    grid = (gain_range, M_range, w_range)

    # Preallocate output
    probs = np.empty(x.shape, dtype=float)

    # For efficiency, build interpolator for all grid points per x using
    # RegularGridInterpolator with vector values by stacking the values along
    # the last dimension.
    # Reshape data to shape (len(gain_range), len(M_range), len(w_range), dimension)
    data = gain_arr

    # Create interpolator that returns a vector of length 'dimension'
    interpolator = RegularGridInterpolator(grid, data, bounds_error=True,method=method)

    # single query point
    point = np.array([gain, M, w])
    vals = interpolator(point)  # returns array of length dimension

    # pick requested x indices
    probs[:] = vals[0,x]

    return probs



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    if True:
        bsm = fits.open('sutin_matrix_output.fits')
        gain_arrays = bsm[0].data[1:, 0:13, 0:13, :]
        gain_range = bsm[1].data[1:13] #[11:12]#[:10]
        M_range = bsm[2].data[:13] #[:10]
        w_range = bsm[3].data[:13] #[:10]
        w = w_range[1] + 4
        M = 300
        g = 7
        #probs = interpolate_Brian_Sutin(g, M, w, np.arange(0,3000), gain_arrays, gain_range, M_range, w_range)
        probs = interpolate_Brian_Sutin_multi(g, M, w, np.arange(0,3000), gain_arrays, gain_range, M_range, w_range, method='slinear')
        #probs2 = interpolate_Brian_Sutin_multi(gain_range[2], 357, w_range[2], np.arange(0,3000), gain_arrays, gain_range, M_range, w_range)
        exact = exact_gain_PDF(g,  M, w, np.arange(0,3000))
        gamma_arr = np.exp(_LogGamma( w, g ,np.arange(0,3000)))

        plt.plot(np.arange(0,3000), probs);plt.plot(np.arange(0,3000), exact); plt.plot(np.arange(0,3000), gamma_arr) 

    #max matrix case is 60k x 60k matrix:  28.8GB; Output of function for this case, though, is just 480kB.
    # Total for all 1000 arrays:  480MB, which is manageable.
    gain_range = np.logspace(0,2.477,20) 
    gain_range = gain_range[1:] # leaving out gain=1
    M_range = np.linspace(50, 900, num=20).astype(int)
    w = np.linspace(1, 100, num=20).astype(int)
    # gain_range = np.logspace(0,2.477,10) #Runs from 10^1 to 10^2.477 = 300 in 10 steps, giving more steps in the lower gain range.  np.linspace(2, 300, num=10) # high gain: Erlang good approximation 
    # M_range = np.linspace(50, 900, num=10).astype(int)
    # w = np.linspace(1, 100, num=10).astype(int)
    # gain_range = np.logspace(0,2,5) #Runs from 10^1 to 10^2.477 = 300 in 10 steps, giving more steps in the lower gain range.  np.linspace(2, 300, num=10) # high gain: Erlang good approximation 
    # M_range = np.linspace(50, 700, num=5).astype(int)
    # w = np.linspace(1, 50, num=5).astype(int)
    dimension = int(gain_range.max() * w.max() * 4) 
    gain_arrays = []
    for gain in gain_range:
        M_arrays = []
        for M in M_range:
            w_arrays = []
            for w_i in w:
                printout = 'gain: ' + str(gain) + ', M: ' + str(M) + ', w:' + str(w_i)
                print('Calculating ' + printout)
                nmax=int(gain*w_i*4)
                if nmax >= 158113: #200 GB matrix; messes up GridInterpolator, but the non _multi interpolator would still work
                    continue
                x_arr = np.arange(0, nmax)
                prob = exact_gain_PDF(gain=gain, M=M, w=w_i, x=x_arr, matrix_thresh=5000, chunk_size=5000)
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

    