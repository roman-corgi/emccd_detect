# -*- coding: utf-8 -*-
"""Generate random numbers according to EM gain pdfs."""

import numpy as np
from scipy import special
from scipy.optimize import fsolve
from partial_CIC_MLE import _LogPn, _LogGamma


class RandEMGainException(Exception):
    """Exception class for rand_em_gain module."""


def rand_em_gain(n_in_array, em_gain):
    """Generate random numbers according to EM gain pdfs.

    Parameters
    ----------
    n_in_array : array_like
        Array of electron values (e-).
    em_gain : float
        EM gain multiplication factor.

    Returns
    -------
    array_like
        Electron values multiplied by random EM gain distribution (e-).

    Notes
    -----
    This function returns an array of the same size as n_in_array. Every element
    in n_in_array is multiplied by em_gain*rand_val, where rand_val is a random
    number drawn from a specific pdf selected based on the value of the
    n_in_array element.

    References
    ----------
    [1] http://matlabtricks.com/post-44/generate-random-numbers-with-a-given-distribution
    [2] https://arxiv.org/pdf/astro-ph/0307305.pdf
    [3] https://arxiv.org/abs/2405.17622

    """
    if em_gain < 1:
        raise RandEMGainException('EM gain cannot be set to less than 1')
    elif em_gain == 1:
        return n_in_array
    else:
        # Apply gain to regular counts
        # Used to do this: Erlang (gamma) distribution variate
        # n_out_array = np.random.gamma(n_in_array, em_gain)
        # n_out_array = np.round(n_out_array)
    
        # more precise method here:
        n_out_array = _apply_gain(n_in_array, em_gain)
        return n_out_array

def _apply_gain(n_in_array, em_gain):
    """Apply a specific em_gain to all nonzero n_in values."""
    # Initialize output count array #XXX Do Pn in array chunks
    n_out_array = np.zeros_like(n_in_array)

    # Get unique nonzero n_in values
    n_in_unique = np.unique(n_in_array)
    n_in_unique = n_in_unique[n_in_unique > 0]

    # Generate random numbers according to the gain distribution for each n_in
    for n_in in n_in_unique:
        inds = np.where(n_in_array == n_in)[0]
        n_out_array[inds] = _rand_pdf(n_in, em_gain, len(inds))

    return np.round(n_out_array)

def _rand_pdf(n_in, em_gain, size):
    """Draw samples from the EM gain distribution."""
    y = np.random.random(size)

    # Use exact solutions for n_in == 1; no exact solution for n_in >= 2 (when summing PDF to get CDF; there is 
    # exact solution for n_in=2 if you integrate PDF to get CDF) 
    if n_in == 1:
        n_out = -em_gain * np.log(1 - y)
    else:
        # For n > 1, sum PDF to get CDF.  Sum up to the value expected 
        # assuming Erlang, and then iterate from there.
        cdfsum = np.ma.masked_array(np.zeros_like(y), mask=np.zeros_like(y).astype(bool))
        n_out = np.zeros_like(y)
        x = np.array([n_in]) # starting term in sum: n = x (lowest x can be is n)
        while cdfsum.mask.min() == 0: 
            cdfsum_prev = cdfsum
            cdfsum = cdfsum + Pn(n_in, em_gain, x)
            mask = (cdfsum > y)
            cdfsum.mask = mask
            keep_inds = np.where(mask == 1)
            # if 1, x-1 closer to y
            preferred_x_1 = np.less(np.abs(cdfsum_prev.data - y), np.abs(cdfsum.data - y)).astype(int)
            # if 1, x closer to y
            preferred_x = 1 - preferred_x_1
            n_out[keep_inds] = preferred_x_1[keep_inds] * (x-1) + preferred_x[keep_inds] * x
            x += 1

    return np.round(n_out)

def CDFErlang_root(x, n, g, y):
    """We use the cumulative distribution function (CDF) for the EM gain 
    probability distribution function (PDF), approximate but appropriate for 
    large EM gain.  This function is used to find the x value when the CDF is 
    equal to y.  The PDF is the Erlang distribution (gamma distribution for 
    integer-valued n), which is normalized when it is integrated.  However, 
    since x should also be integer-valued, the sum should be used instead.  
    In that case, there is a normalization factor dependent on g and n, but the 
    factor is very close to 1 in general.  Since we are already approximating
    by using this PDF, we leave off this troublesome factor, which is in terms
    of the Hurwitz-Lerch transcendent. 
    
    Parameters
    ----------
    x : array-like (1D)
        PDF variate values, one for each value in n array.
    n : array-like (1D)
        Input array of electron values (e-) before entering the gain register.
    g: float
        EM gain value.
    y : float
        Desired CDF value.

    Returns
    -------
    array-like (1D)
        CDF values.
    """
    CDF = 1 - special.gammaincc(n, x/g)
    return CDF - y

def Pn(n, g, x):
    """The probability distribution function (PDF) for the normalized EM gain, 
    valid for small and large gain values.
    
    Parameters
    ----------
    n : array-like (1D)
        Input array of electron values (e-) before entering the gain register.
    g: float
        EM gain value.
    x : array-like (1D)
        PDF variate values, one for each value in n array.

    Returns
    -------
    array-like (1D)
        PDF values.

    """
    out = _LogPn(n, g, x)
    Pn = np.exp(out)
    return Pn


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    x_arr = np.arange(0, 200)
    plt.plot(x_arr, np.exp(_LogGamma(100,2,x_arr)))
    plt.plot(x_arr, Pn(100,2,x_arr))

    output = _rand_pdf(3,4,3)

    # Generally, the agreement b/w the old and new methods is good.  The new
    # method just speeds up the code a lot, especially when cosmics are present.
    # Old method functions below:

    def _apply_gain_old(n_in_array, em_gain, max_out):
        """Apply a specific em_gain to all nonzero n_in values."""
        # Initialize output count array
        n_out_array = np.zeros_like(n_in_array)

        # Get unique nonzero n_in values
        n_in_unique = np.unique(n_in_array)
        n_in_unique = n_in_unique[n_in_unique > 0]

        # Generate random numbers according to the gain distribution for each n_in
        for n_in in n_in_unique:
            inds = np.where(n_in_array == n_in)[0]
            n_out_array[inds] = _rand_pdf_old(n_in, em_gain, max_out, len(inds))

        # n_out_array = np.random.gamma(n_in_array, em_gain)
        # n_out_array = np.round(n_out_array)

        return n_out_array


    def _rand_pdf_old(n_in, em_gain, x_max, size):
        """Draw samples from the EM gain distribution."""
        x = np.random.random(size)

        # Use exact solutions for n_in == 1 and 2
        if n_in == 1:
            n_out = -em_gain * np.log(1 - x)
        elif n_in == 2:
            n_out = -em_gain * special.lambertw((x-1)/np.exp(1), -1).real - em_gain
        else:
            # For n > 2 use CDF approximation
            # Use x values ranging from 0 to maximum allowable x output
            x_axis = np.arange(0, x_max).astype(float)
            x_axis[0] = np.finfo(float).eps  # Use epsilon to avoid divide by 0
            cdf = _get_cdf(n_in, em_gain, x_axis)

            if cdf is None:
                # If cdf maxes out, return maximum value
                n_out = np.ones_like(x) * x_max
            else:
                # Draw random samples from the CDF
                cdf_lookups = (cdf.max() - cdf.min()) * x + cdf.min()
                n_out = x_axis[np.searchsorted(cdf, cdf_lookups)]  # XXX This could be made more accurate

        return np.round(n_out)


    def _get_cdf(n_in, em_gain, x):
        """Return an approximate CDF for the EM gain distribution.

        Basden 2003 probability distribution function is as follows:

            pdf = x^(n_in-1) * exp(-x/g) / (g^n_in * factorial(n_in-1))

        """
        # Because of the cancellation of very large numbers, first work in log space
        logpdf = (n_in-1)*np.log(x) - x/em_gain - n_in*np.log(em_gain) - special.gammaln(n_in)
        pdf = np.exp(logpdf)

        # XXX This is a rough but safe solution
        sum_pdf = np.sum(pdf)
        if sum_pdf == 0:
            cdf = None
        else:
            cdf = np.cumsum(pdf / sum_pdf)

        return cdf

    def compare_stats(g, n, n_samples, max_val, num_bins, plot=False):

        n_in_array = np.array([n]*n_samples)
        old_method = _apply_gain_old(n_in_array, g, max_val)

        # gamma distribution:
        x = rand_em_gain(n_in_array, g)

        print("For n={}, g={}:".format(n,g))
        print('Mean for old method:  ', np.mean(old_method))
        print('Std dev for old method:  ', np.std(old_method))
        print('Mean of gamma distribution:  ', np.mean(x))
        print('Std dev for gamma distribution:  ', np.std(x))
        print('theoretical mean:  ', g*n)
        print('theortical std dev:  ', g*np.sqrt(n))
        print()

        if plot==True:
            fig, ax = plt.subplots()
            H = ax.hist(old_method, bins = num_bins)
            ax.set_ylabel('number of occurrences')
            ax.set_xlabel('gained counts (e-)')
            ax.set_title('Histogram of Gained Counts (Old Method, n={})'.format(n))

            fig, ax = plt.subplots()
            H = ax.hist(x, bins = num_bins)
            ax.set_ylabel('number of occurrences')
            ax.set_xlabel('gained counts (e-)')
            ax.set_title('Histogram of Gained Counts (Gamma Distribution, n={})'.format(n))

            plt.show()

    g = 2 #20 #200
    n_samples = 10000
    #max_val = 200000
    num_bins = 40
    # let max_out be the mean + 4*std dev from gamma dist for the max value
    # from serial_counts (using ENF ~ sqrt(2), which is fine even for low
    # gain since we just want an upper limit)
    def max_val(g, n):
        return g*n + 4*g*np.sqrt(2*n)

    # in original code, max_val used max(n_in_array) where that array was for
    # all serial cells; so artifically inflate by multiplying by 100
    n = 1
    compare_stats(g, n, n_samples, 100*max_val(g,n), num_bins)

    n2 = 2
    compare_stats(g, n2, n_samples, 100*max_val(g,n), num_bins)

    # now a value of n for which these methods differed
    n3 = 3
    compare_stats(g, n3, n_samples, 100*max_val(g,n), num_bins)

    n4 = 40
    compare_stats(g, n4, n_samples, 100*max_val(g,n), num_bins)

    n5 = 100
    compare_stats(g, n5, n_samples, 100*max_val(g,n), num_bins)
