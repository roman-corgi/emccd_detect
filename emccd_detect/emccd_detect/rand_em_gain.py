# -*- coding: utf-8 -*-
"""Generate random numbers according to EM gain pdfs."""

import numpy as np
from scipy import special
from emccd_detect.partial_CIC_MLE import _LogPn, _LogGamma
#from partial_CIC_MLE import _LogPn, _LogGamma


class RandEMGainException(Exception):
    """Exception class for rand_em_gain module."""


def rand_em_gain(n_in_array, em_gain, quick, threshold):
    """Generate random numbers according to EM gain pdfs.

    Parameters
    ----------
    n_in_array : array_like
        Array of electron values (e-).
    em_gain : float
        EM gain multiplication factor.
    threshold : float
        Threshold for switching between methods.  If the product of n_in and 
        the size of the array is greater than this threshold, the faster, less memory-intensive method 
        (gamma distribution) is used.  Otherwise, the more accurate method (Pn) is used.

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
        n_out_array = _apply_gain(n_in_array, em_gain, quick, threshold)
        return n_out_array

def _apply_gain(n_in_array, em_gain, quick, threshold):
    """Apply a specific em_gain to all nonzero n_in values."""
    # Initialize output count array
    n_out_array = np.zeros_like(n_in_array)

    if quick:
        n_out_array = np.random.gamma(n_in_array, em_gain)
    else:
        # need integer values for Pn().  Otherwise, we try to keep e- values as floats since EM gain,
        # k gain, nonlinearity, master flat, etc are calibrated assuming fractions of electrons, and 
        # we can get fractions of electrons here for particular gain values.  So we just round 
        # DN output to integer at the end.
        n_in_array = np.round(n_in_array) 

        gamma_inds = np.where(n_in_array*(n_in_array*em_gain - n_in_array) >= threshold)
        not_gamma_inds = np.where(n_in_array*(n_in_array*em_gain - n_in_array) < threshold)
        n_out_array[gamma_inds] = np.random.gamma(n_in_array[gamma_inds], em_gain)
        # For the others, get unique nonzero n_in values
        n_in_unique = np.unique(n_in_array[not_gamma_inds])

        if n_in_unique.size != 0:
            n_in_unique = n_in_unique[n_in_unique > 0]
            # Generate random numbers according to the gain distribution for each n_in
            for n_in in n_in_unique:                
                inds = np.where(n_in_array == n_in)[0]
                n_out_array[inds] = _rand_pdf(int(np.round(n_in)), em_gain, len(inds), threshold=threshold)

    return np.round(n_out_array)

def _rand_pdf(n_in, em_gain, size, threshold=1e7):
    """Draw samples from the EM gain distribution."""
    y = np.random.random(size) # ranges from [0,1)

    # Use exact solutions for n_in == 1; no exact solution for n_in >= 2 (when summing PDF to get CDF; there is 
    # exact solution for n_in=2 if you integrate PDF to get CDF) 
    if n_in == 1:
        n_out = -em_gain * np.log(1 - y)
    else:
        # For n > 1, sum PDF to get CDF. 
        n_out = np.ones_like(y).astype(float)*np.nan
        # as a good initial guess, generate up to n_in*em_gain*y/0.5, which is the Erlang mean scaled by y relative to 0.5 prob, where the mean is 
        x = np.arange(n_in, np.round(n_in*em_gain*y.max()/0.5))
        if n_in * x.size >= threshold: #was 2e8 originally 
            #revert back to gamma distribution, valid for large x values. 
            # When n*g is large, the mean will be large, and most of the non-zero CDF
            # happens around there, but if a small y value is requested, it could 
            # be that the corresponding x is small, so this condition is a little 
            # better than using gamma distribution when n*g is large. We consider
            # n_in*x.size b/c that is the practical memory bottleneck which can stop the program.
            n_out = np.random.gamma(n_in, em_gain, size=size) # then the while loop will be skipped, and this n_out will be returned
        cdfsum = np.array([0]) #initialize 
        counter = 0
        while np.where(np.isnan(n_out))[0].size > 0: 
            cdfsum = np.append(cdfsum[-1], cdfsum[-1] + np.cumsum(Pn(n_in, em_gain, x)))
            x = np.append(x[0]-1, x) # add one more term to sum to match cdfsum
            # find where y values should go in cdfsum array
            keep_inds = np.where(y <= cdfsum[-1])
            keep_inds = np.intersect1d(keep_inds, np.where(y >= cdfsum[0]))
            cdf_inds = np.searchsorted(cdfsum.data, y[keep_inds])
            cdf_inds2 = np.where(cdf_inds == 0, 1, cdf_inds) # to avoid out of bounds error when cdf_inds is 0; in this case where 0 is in cdf_inds, both options below are the same
            # if 1, x[cdf_inds] closer to y
            preferred_x = np.less(np.abs(cdfsum[cdf_inds] - y[keep_inds]), np.abs(cdfsum[cdf_inds2-1] - y[keep_inds])).astype(int)
            # if 1, x[cdf_inds-1] closer to y
            preferred_x_1 = 1 - preferred_x
            n_out[keep_inds] = preferred_x_1 * x[cdf_inds2-1] + preferred_x * x[cdf_inds]
            if counter == 0:
                n_out[np.where(y < cdfsum[0])] = n_in 
            x = np.arange(x[-1]+1, 2*(x[-1]+1)) # increase x range if needed 
            counter += 1

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
    import matplotlib.pyplot as plt

    # x_arr = np.arange(0, 200)
    # plt.plot(x_arr, np.exp(_LogGamma(100,2,x_arr)))
    # plt.plot(x_arr, Pn(100,2,x_arr))

    n=100;g=1000
    x_arr = np.round(np.linspace(n, n*g*4, num=4000)).astype(int)
    plt.plot(x_arr, np.exp(_LogGamma(n,g,x_arr)))
    plt.plot(x_arr, Pn(n,g,x_arr))

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
        old_method = rand_em_gain(n_in_array, g, quick=True, threshold=1e7) 
        #old_method = _apply_gain_old(n_in_array, g, max_val)

        # new method:
        x = rand_em_gain(n_in_array, g, quick=False, threshold=1e7)

        print("For n={}, g={}:".format(n,g))
        print('Mean for old method:  ', np.mean(old_method))
        print('Std dev for old method:  ', np.std(old_method))
        print('Mean of new method:  ', np.mean(x))
        print('Std dev for new method:  ', np.std(x))
        print('Difference of means:  ', np.mean(old_method) - np.mean(x))
        print('Percentage of std dev for difference:  ', (np.mean(old_method) - np.mean(x))/(g*n))
        print('theoretical mean (gamma):  ', g*n)
        print('theortical std dev (gamma):  ', g*np.sqrt(n))
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
            ax.set_title('Histogram of Gained Counts (New Method, n={})'.format(n))

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
    np.random.seed(123) # for reproducibility
    compare_stats(g, n, n_samples, 100*max_val(g,n), num_bins)

    n2 = 2
    np.random.seed(123) # for reproducibility
    compare_stats(g, n2, n_samples, 100*max_val(g,n), num_bins)

    # now a value of n for which these methods differed
    n3 = 3
    np.random.seed(123) # for reproducibility
    compare_stats(g, n3, n_samples, 100*max_val(g,n), num_bins)

    n4 = 40
    np.random.seed(123) # for reproducibility
    compare_stats(g, n4, n_samples, 100*max_val(g,n), num_bins)

    n5 = 100
    np.random.seed(123) # for reproducibility
    compare_stats(g, n5, n_samples, 100*max_val(g,n), num_bins)

    n6 = 100
    g6 = 100
    np.random.seed(123) # for reproducibility
    compare_stats(g6, n6, n_samples, 100*max_val(g6,n6), num_bins)

    n7 = 1000
    g7 = 1000
    np.random.seed(123) # for reproducibility
    compare_stats(g7, n7, n_samples, 100*max_val(g7,n7), num_bins)

    n8 = 18000
    g8 = 5000
    np.random.seed(123) # for reproducibility
    compare_stats(g8, n8, n_samples, 100*max_val(g8,n8), num_bins)