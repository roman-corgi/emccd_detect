# -*- coding: utf-8 -*-
"""Take a moving mean over a vector."""
import numpy as np


class MovingMeanException(Exception):
    """Exception class for moving_mean module."""
    pass


def movmean(a, k):
    """Take a moving mean over a vector.

    Paremeters
    ----------
    a : array_like
        Array with ndim of one.
    k : int
        Window size.

    Returns
    -------
    moving : array_like, float
        Averaged array, same size as input array.

    Notes
    -----
    This is meant to loosely replicate Matlab's movmean. Odd windows (k) are
    centered on the element in the current position, and even windows are
    centered about the current element and the previous element. The window
    size is truncated at the endpoints and the mean is only taken over the
    points that fill the window.

    S Miller - UAH - 6-May-2019
    """
    try:
        a_array = np.array(a).astype(float)
        if a_array.ndim != 1:
            raise MovingMeanException('Input array must be one dimensional')
    except Exception:
        raise MovingMeanException('Error converting to numpy array')

    if k <= 0 or not isinstance(k, int):
        raise MovingMeanException('Window size must be a positive integer')
    if k > len(a_array):
        raise MovingMeanException('Window size is larger than array')

    # Take moving sum
    moving_sum = a_array.cumsum()
    moving_sum[k:] = moving_sum[k:] - moving_sum[:-k]

    # Take mean of beginning and end of vector (where window is truncated)
    half = np.ceil(k / 2.).astype(int)
    moving = moving_sum[half-1:]
    moving = _truncate_beg(moving, half)
    end = _truncate_end(a_array, k, half)

    # Take mean of the rest of the vector
    moving[half:] = moving[half:] / k
    moving = np.append(moving, end)

    return moving


def _truncate_beg(moving, half):
    """Calculate mean for truncated window at beginning of vector.

    Parameters
    ----------
    moving : array_like, float
        Moving mean array in progress.
    half : int
        Window size ceiling divided by two, used for truncating.

    Return
    ------
    moving : array_like, float
        Moving mean array in progress.
    """
    for i in range(half):
        k_trunc = half + i
        moving[i] = moving[i] / k_trunc

    return moving


def _truncate_end(a_array, k, half):
    """Calculate mean for truncated window at end of vector.

    Parameters
    ----------
    a_array : array_like, float
        Input array.
    k : int
        Window size.
    half : int
        Window size ceiling divided by two, used for truncating.

    Return
    ------
    end : array_like, float
        End of moving mean array.
    """
    end = [0] * (half-1)
    for i in range(half-1):
        k_trunc = k-1 - i
        end[i] = np.sum(a_array[-k_trunc:]) / k_trunc

    return end
