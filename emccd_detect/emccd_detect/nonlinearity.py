# -*- coding: utf-8 -*-
"""Module for imposing nonlinearity (inverse of relative gains) for each 
pixel of a frame.

Relative gain is dependent on both the detector gain and the dn count
value of a given pixel.

Adapted from https://github.com/roman-corgi/cgi_iit_drp/blob/main/proc_cgi_frame_NTR/proc_cgi_frame/gsw_nonlin.py.
"""

import numpy as np
from scipy import interpolate


class NonlinException(Exception):
    """Exception class for nonlin module."""


def _parse_file(nonlin_path):
    """Get data from nonlinearity correction file."""
    # Read nonlin csv
    nonlin_raw = np.genfromtxt(nonlin_path, delimiter=',')

    # File format checks
    if nonlin_raw.ndim < 2 or nonlin_raw.shape[0] < 2 or \
       nonlin_raw.shape[1] < 2:
        raise NonlinException('Nonlin array must be at least 2x2 (room for x '
                              'and y axes and one data point)')
    if not np.isnan(nonlin_raw[0, 0]):
        raise NonlinException('First value of csv (upper left) must be set to '
                              '"nan"')

    # Column headers are gains, row headers are dn counts
    gain_ax = nonlin_raw[0, 1:]
    count_ax = nonlin_raw[1:, 0]
    # Array is relative gain values at a given dn count and gain
    relgains = nonlin_raw[1:, 1:]

    # Check for increasing axes
    if np.any(np.diff(gain_ax) <= 0):
        raise NonlinException('Gain axis (column headers) must be increasing')
    if np.any(np.diff(count_ax) <= 0):
        raise NonlinException('Counts axis (row headers) must be increasing')
    # Check that curves (data in columns) contain or straddle 1.0
    if (np.min(relgains, axis=0) > 1).any() or \
       (np.max(relgains, axis=0) < 1).any():
        raise NonlinException('Gain curves (array columns) must contain or '
                              'straddle a relative gain of 1.0')

    return gain_ax, count_ax, relgains


def apply_relgains(frame, em_gain, nonlin_path):
    """For a given bias-subtracted flattened frame of dn counts, 
    return a same-size array of inverse relative gain values.  
    "Relative gain" here is the value 
    to correct for nonlinearity, the values from the input at nonlin_path, but 
    since this function applies nonlinearity, it returns the corresponding 
    inverse values.

    The required format for the file specified at nonlin_path is as follows:
     - CSV
     - Minimum 2x2
     - First value (top left) must be assigned to nan
     - Row headers (dn counts) must be monotonically increasing
     - Column headers (EM gains) must be monotonically increasing
     - Data columns (relative gain curves) must straddle 1

    For example:

    [
        [nan,  1,     10,    100,   1000 ],
        [1,    0.900, 0.950, 0.989, 1.000],
        [1000, 0.910, 0.960, 0.990, 1.010],
        [2000, 0.950, 1.000, 1.010, 1.050],
        [3000, 1.000, 1.001, 1.011, 1.060],
    ],

    where the row headers [1, 1000, 2000, 3000] are dn counts, the column
    headers [1, 10, 100, 1000] are EM gains, and the first data column
    [0.900, 0.910, 0.950, 1.000] is the first of the four relative gain curves.

    Parameters
    ----------
    frame : array_like
        Flattened array of dn count values.
    em_gain : float
        Detector EM gain.
    nonlin_path : str
        Full path of nonlinearity calibration csv.

    Returns
    -------
    array_like
        Flattened array of inverse relative gain values.

    Notes
    -----
    This algorithm contains two interpolations:

     - A 2d interpolation to find the relative gain curve for a given EM gain
     - A 1d interpolation to find a relative gain value for each given dn
     count value.

    Both of these interpolations are linear, and both use their edge values as
    constant extrapolations for out of bounds values.

    """

    # Get file data
    gain_ax, count_ax, relgains = _parse_file(nonlin_path)

    # Create interpolation for em gain (x), counts (y), and relative gain (z).
    # Note that this defaults to using the edge values as fill_value for
    # out of bounds values (same as specified below in interp1d)
    f = interpolate.RectBivariateSpline(gain_ax,
                                    count_ax,
                                    relgains.T,
                                    kx=1,
                                    ky=1,
    )
    # Get the relative gain curve for the given gain value
    relgain_curve = f(em_gain, count_ax)[0]

    # Create interpolation for dn counts (x) and relative gains (y). For
    # out of bounds values use edge values
    ff = interpolate.interp1d(count_ax, relgain_curve, kind='linear',
                              bounds_error=False,
                              fill_value=(relgain_curve[0], relgain_curve[-1]))
    # For each dn count, find the inverse of the relative gain since
    # we are applying nonlinearity instead of correcting for it
    counts_flat = 1/ff(frame)

    return counts_flat