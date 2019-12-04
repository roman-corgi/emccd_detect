# -*- coding: utf-8 -*-
"""Utilities for working in polar coordinates."""
import numpy as np


class RadAvgException(Exception):
    """Exception class for rad_avg module."""
    pass


def cart2pol(x, y):
    """Convert from cartesian to polar coordinates.

    Parameters
    ----------
    x : array_like
        Meshgrid array of x coordinates.
    y : array_like
        Meshgrid array of y coordinates.

    Returns
    -------
    theta : array_like
        Meshgrid array of theta coordinates.
    rho : array_like
        Meshgrid array of rho coordinates.

    Notes
    -----
    This is meant to loosely replicate Matlab's cart2pol function.

    S Miller - UAH - 25-Jul-2019
    """
    theta = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)

    return(theta, rho)


def rad_avg(a, step=1):
    """Take radial averages of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    step : float, optional
        Step size of radial slices. Defaults to one.

    Returns
    -------
    avg : array_like
        One dimensional array of radial averages.
    steps : array_like
        Radial coordinates of steps corresponding to averages.

    Notes
    -----
    Note that the input array is assumed to be a square, nxn array. It is also
    assumed to be centered at (n-1)/2 (if n is odd) or n/2 (if n is even).

    S Miller - UAH - 25-Jul-2019
    """
    n, m = a.shape
    if n != m:
        raise RadAvgException('Input array must be square')

    if (n % 2) == 0:
        half = n/2
    else:
        half = (n-1)/2
    x = np.arange(-half, half+1)

    xx, yy = np.meshgrid(x, x)
    theta, rho = cart2pol(xx, yy)
    steps = np.arange(0, x[-1]+1, step)

    avg = np.zeros(len(steps))
    n = 0
    for i in steps:
        rad = a[np.logical_and(rho <= i, rho > i-step)]
        avg[n] = np.mean(rad)
        n += 1

    return (avg, steps)


def rad_std(a, step=1):
    """Take radial standard deviation of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    step : float, optional
        Step size of radial slices. Defaults to one.

    Returns
    -------
    std : array_like
        One dimensional array of radial standard deviations.
    steps : array_like
        Radial coordinates of steps corresponding to averages.

    Notes
    -----
    Note that the input array is assumed to be a square, nxn array. It is also
    assumed to be centered at (n-1)/2 (if n is odd) or n/2 (if n is even).

    S Miller - UAH - 26-Jul-2019
    """
    n, m = a.shape
    if n != m:
        raise RadAvgException('Input array must be square')

    if (n % 2) == 0:
        half = n/2
    else:
        half = (n-1)/2
    x = np.arange(-half, half+1)

    xx, yy = np.meshgrid(x, x)
    theta, rho = cart2pol(xx, yy)
    steps = np.arange(0, x[-1]+1, step)

    std = np.zeros(len(steps))
    n = 0
    for i in steps:
        rad = a[np.logical_and(rho <= i, rho > i-step)]
        std[n] = np.std(rad)
        n += 1

    return (std, steps)
