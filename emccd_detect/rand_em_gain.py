"""Generate random numbers according to EM gain pdfs."""
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy import special


def rand_em_gain(n_in_array, em_gain):
    """Generate random numbers according to EM gain pdfs.

    References
    ----------
    [1] http://matlabtricks.com/post-44/generate-random-numbers-with-a-given-distribution
    [2] https://arxiv.org/pdf/astro-ph/0307305.pdf

    B Nemati and S Miller - UAH - 20-March-2020

    """

    if em_gain < 1:
        raise Exception('EM gain cannot be set to less than 1')

    # Find how many values in a array are equal to 0, 1, 2, or >= 3
    y = np.zeros(n_in_array.size, 1)
    inds0 = (n_in_array == 0).nonzero()
    inds1 = (n_in_array == 1).nonzero()
    inds2 = (n_in_array == 2).nonzero()
    inds3 = (n_in_array > 2).nonzero()

    # For n_in of 0, 1, or 2, generate arrays of random numbers according to gain
    # equations specific to each n_in
    n0 = len(inds0)
    n1 = len(inds1)
    n2 = len(inds2)
    y[inds0] = rand_em_exact(0, n0, em_gain)
    y[inds1] = rand_em_exact(1, n1, em_gain)
    y[inds2] = rand_em_exact(2, n2, em_gain)

    # For n_in of 3 or greater, generate random numbers one by one according to the
    # generalized gain equation
    for i in inds3:
        n_in = n_in_array(i)
        y[i] = rand_em_approx(n_in, em_gain)

    return np.reshape(y, n_in_array.shape)


def rand_em_exact(n_in, numel, g):
    """Select a gain distribution based on n_in and generate random numbers."""
    x = np.ranodm.random(numel, 1)

    if n_in == 0:
        y = np.zeros(numel, 1)
    elif n_in == 1:
        y = -g * np.log(1 - x)
    elif n_in == 2:
        y = -g * special.lambertw(-1, (x-1)/np.exp(1)) - g

    return np.round(y)


def rand_em_approx(n_in, g):
    """Select a gain distribution based on n_in and generate a single random number."""
    kmax = 5
    xmin = np.finfo(float).eps
    xmax = kmax * n_in * g
    nx = 800
    x = np.linspace(xmin, xmax, nx)  # XXX Check against old version

    # Basden 2003 probability distribution function is as follows:
    # pdf = x.^(n_in-1) .* exp(-x/g) / (g^n_in * factorial(n_in-1))
    # Because of the cancellation of very large numbers, first work in log space
    logpdf = (n_in-1)*np.log(x) - x/g - n_in*np.log(g) - special.gammaln(n_in)
    pdf = np.exp(logpdf)
    cdf = np.cumsum(pdf / np.sum(pdf))

    # Create a uniformly distributed random number for lookup in CDF
    cdf_lookup = np.random.uniform(min(cdf), max(cdf))

    # Map random value to cdf
    ihi = (cdf > cdf_lookup).nonzero()[0]
    ilo = ihi - 1
    xlo = x[ilo]
    xhi = x[ihi]
    clo = cdf[ilo]
    chi = cdf[ihi]
    y = xlo + (cdf_lookup - clo) * (xhi - xlo)/(chi-clo)

    return np.round(y)
