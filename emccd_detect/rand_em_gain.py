# -*- coding: utf-8 -*-
"""Generate random numbers according to EM gain pdfs."""
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy import special


class RandEMGainException(Exception):
    """Exception class for rand_em_gain module."""


def rand_em_gain(n_in_array, em_gain, max_out):
    """Generate random numbers according to EM gain pdfs.

    Parameters
    ----------
    n_in_array : array_like
        Array of electron values (e-).
    em_gain : float
        CCD em_gain (e-/photon).
    max_out : float
        Maximum allowed output, used to set a end bound on distributions (e-).

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

    B Nemati and S Miller - UAH - 20-March-2020

    """
    if em_gain < 1:
        raise RandEMGainException('EM gain cannot be set to less than 1')

    # Initialize output count array
    n_out_array = np.zeros_like(n_in_array)

    # Get unique nonzero n_in values
    n_in_unique = np.unique(n_in_array)
    n_in_unique = n_in_unique[n_in_unique > 0]

    # Generate random numbers according to the gain distribution for each n_in
    for n_in in n_in_unique:
        inds = np.where(n_in_array == n_in)[0]
        n_out_array[inds] = _rand_pdf(n_in, em_gain, max_out, len(inds))

    return n_out_array


def _rand_pdf(n_in, em_gain, x_max, size):
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
    cdf = np.cumsum(pdf / np.sum(pdf))

    return cdf
