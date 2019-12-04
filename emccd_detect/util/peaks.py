# -*- coding: utf-8 -*-
import numpy as np


def peaks(n=49):
    """Return example function of two variables.

    Parameters
    ----------
    n : int, optional
        Size of nxn output array. Defaults to 49.

    Returns
    -------
    z : ndarray
        Output array.

    Notes
    -----
    This is meant to replicate Matlab's peaks function.

    S Miller - UAH - 7-Feb-2019
    """
    xx = np.linspace(-3, 3, n)
    yy = np.linspace(-3, 3, n)
    [x, y] = np.meshgrid(xx, yy)
    z = (3 * (1-x)**2 * np.exp(-(x**2) - (y+1)**2)
         - 10*(x/5.0 - x**3 - y**5) * np.exp(-x**2 - y**2)
         - 1/3.0*np.exp(-(x+1)**2 - y**2))

    return z
