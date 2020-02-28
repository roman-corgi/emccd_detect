# -*- coding: utf-8 -*-
import numpy as np
from scipy import special


def rand_em_gain(n_in, em_gain):
    """Generate random number according to the EM gain prob density function.

    Parameters
    ----------
    n_in : float
        Number of electrons entering EM register.
    em_gain : float
        Mean electron multiplication gain of the gain register.

    Returns
    -------
    n_out : float
        Number of electrons exiting EM register.

    Notes
    -----
    This is an approximate, fast algorithm [1]_. The mean of the random
    variates created is about 1% low systematically, but for photon counting it
    should be a next-order effect and small.

    See Basden 2003 paper [2]_.

    .. [1] http://matlabtricks.com/post-44/generate-random-numbers-with-a-given-distribution
    .. [2] https://arxiv.org/pdf/astro-ph/0307305.pdf

    B Nemati and S Miller - UAH - 22-Sep-2018
    """
    if em_gain < 1:
        raise ValueError('EM gain cannot be set to less than 1')

    if n_in < 16:
        kmax = 4
        xmin = np.finfo(float).eps
        xmax = kmax * n_in * em_gain
        xcorr = 0.5
        if n_in < 3:
            em_gamma = 0
        else:
            em_gamma = special.gammaln(n_in)
    else:
        kmax = 4
        xmin = (n_in - kmax*np.sqrt(n_in)) * em_gain
        xmax = (n_in + kmax*np.sqrt(n_in)) * em_gain
        xcorr = 0.3
        em_gamma = special.gammaln(n_in)

    x = np.arange(xmin, xmax+(xmax-xmin)/99, (xmax-xmin)/99)

    if n_in == 1:
        xn_in = 0
    else:
        xn_in = (n_in-1) * np.log(x+1)

    # Basden 2003 probability distribution function
    # The probability distribution function is as follows:
    # pdf = x^(n_in-1) * exp(-x/em_gain) / (em_gain^n_in * factorial(n_in-1))
    # Because of the cancellation of very large numbers, first work in log
    # space
    logpdf = xn_in - x/em_gain - n_in*np.log(em_gain) - em_gamma
    pdf = np.exp(logpdf)

    # A very ad-hoc correction: compensate for chopped off high tail by skewing
    # the pdf
    corr_skew = 1 + xcorr * np.arange(0, 1+1.0/(len(x)-1), 1.0/(len(x)-1))
    pdf = pdf * corr_skew

    # Generate random numbers according to pdf
    pdf = pdf / sum(pdf)
    cdf = np.cumsum(pdf)

    # Create a uniformly distributed random number for lookup in CDF
    cdf_lookup = np.random.random(1)

    if cdf_lookup < cdf[0]:
        randout = 0
    else:
        ihi = (cdf > cdf_lookup).nonzero()[0][0]
        ilo = ihi - 1
        xlo = x[ilo]
        xhi = x[ihi]
        clo = cdf[ilo]
        chi = cdf[ihi]
        randout = xlo + (cdf_lookup - clo) * ((xhi - xlo)/(chi-clo))

    n_out = round(randout)

    return n_out
