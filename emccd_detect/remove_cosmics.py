# -*- coding: utf-8 -*-
"""Remove cosmics from bias subtracted image area of frame."""
import copy

import numpy as np
from scipy.ndimage import median_filter

from util.moving_mean import movmean


class RemoveCosmicsException(Exception):
    """Exception class for remove_cosmics module."""
    pass


def remove_cosmics(bs_image, fwc, sat_thresh, plat_thresh, cosm_filter, tail_filter):
    """Identify and remove cosmic ray hits and tails.

    Use sat_thresh (interval 0 to 1) to set the threshold above which cosmics
    will be detected. For example, sat_thresh=0.99 will detect cosmics above
    0.99*fwc.

    Use plat_thresh (interval 0 to 1) to set the threshold under which cosmic
    plateaus will end. For example, if plat_thresh=0.85, once a cosmic is
    detected the beginning and end of its plateau will be determined where the
    pixel values drop below 0.85*fwc.

    Use cosm_filter to determine the smallest plateaus (in pixels) that will
    be identified. A reasonable value is 3.

    Use tail_filter to determine the window size for the moving median that
    smooths the cosmic tail. A reasonable value is 5.

    Parameters
    ----------
    bs_image : array_like, float
        Bias subtracted image area of frame.
    fwc : float
        Full well capacity of detector.
    sat_thresh : float
        Multiplication factor for fwc that determines saturated cosmic pixels.
    plat_thresh : float
        Multiplication factor for fwc that determines edges of cosmic plateu.
    cosm_filter : int
        Minimum length in pixels of cosmic plateus to be identified.
    tail_filter : int
        Moving median filter window size for cosmic tail subtraction.

    Returns
    -------
    cleaned : array_like, float
        Bias subtracted image area of frame with cosmics removed.
    cosm_mask : array_like, int
        Mask for cosmic plateau pixels that have been set to zero.
    tail_mask : array_like, int
        Mask for cosmic tails that have been cleaned using a moving median.

    Notes
    -----
    This algorithm uses a row by row method for cosmic removal. It first finds
    streak rows, which are rows that potentially contain cosmics. It then
    filters each of these rows in order to differentiate cosmic hits (plateaus)
    from any outlier saturated pixels. For each cosmic hit it finds the edges
    of the plateaus, sets the plateau to zero, and subtracts a moving median
    from the tail.

    |<--------- streak row is the whole row ----------------------->|
     ......|<-platteau->|<------------------tail------------------->|

    B Nemati and S Miller - UAH - 02-Oct-2018
    """
    cleaned = copy.copy(bs_image)
    cosm_mask = np.zeros(bs_image.shape, dtype=int)
    tail_mask = np.zeros(bs_image.shape, dtype=int)
    i_streak_rows = find_cosmic_rows(bs_image, fwc, sat_thresh)

    for i in i_streak_rows:
        row = cleaned[i]
        c_mask_row = cosm_mask[i]
        t_mask_row = tail_mask[i]
        # Find if and where saturated plateaus are located in streak row
        i_begs, i_ends = find_plateaus(row, fwc, sat_thresh, plat_thresh,
                                       cosm_filter)
        # If plateaus exist, correct for cosmic hits and tails
        if len(i_begs) > 0:
            cleaned[i], cosm_mask[i], tail_mask[i] = clean_row(row, c_mask_row,
                                                               t_mask_row,
                                                               i_begs, i_ends,
                                                               tail_filter)

    return cleaned, cosm_mask, tail_mask


def find_cosmic_rows(bs_image, fwc, sat_thresh):
    """Find rows that may contain cosmic hits.

    Parameters
    ----------
    bs_image : array_like, float
        Bias subtracted image area of frame.
    fwc : float
        Full well capacity of detector.
    sat_thresh : float
        Multiplication factor for fwc that determines saturated cosmic pixels.

    Returns
    -------
    i_streak_rows : array_like, float
        Indices of rows which may contain cosmics.
    """
    max_rows = np.max(bs_image, axis=1)
    i_streak_rows = (max_rows > sat_thresh*fwc).nonzero()[0]

    return i_streak_rows


def find_plateaus(streak_row, fwc, sat_thresh, plat_thresh, cosm_filter):
    """Find the beginning and end indices of each cosmic plateau in a row.

    Note that i_begs and i_ends are set at one pixel before and past the last
    plateau pixels, respectively, as these pixels immediately neighboring the
    cosmic plateau are very often affected by the cosmic hit as well.

    Parameters
    ----------
    streak_row : array_like, float
        Row with possible cosmics.
    fwc : float
        Full well capacity of detector.
    sat_thresh : float
        Multiplication factor for fwc that determines saturated cosmic pixels.
    plat_thresh : float
        Multiplication factor for fwc that determines edges of cosmic plateu.
    cosm_filter : int
        Minimum length in pixels of cosmic plateus to be identified.

    Returns
    -------
    i_begs : array_like, int
        Indices of plateau beginnings.
    i_ends : array_like, int
        Indices of plateau endings.
    """
    # Lowpass filter row to differentiate plateaus from standalone pixels
    # XXX Might want to use median filter here, mean doesn't handle uneven
    # plateaus well (ignores plateaus that shouldn't be ignored)
    filtered = movmean(streak_row, cosm_filter)
    saturated = (filtered >= sat_thresh*fwc).nonzero()[0]
    i_begs = np.array([], dtype=int)
    i_ends = np.array([], dtype=int)

    if saturated.any():
        # Find distance between plateaus of saturated pixels
        diff = np.diff(saturated)
        jump = (diff > 1).nonzero()[0]
        jump = np.concatenate(([0], jump+1, [len(saturated)]))

        # For each plateau, find the beginning and end
        n_plats = len(jump) - 1
        for i in range(n_plats):
            i_beg = saturated[jump[i]]
            i_end = saturated[jump[i+1]-1]
            while i_beg > 0 and streak_row[i_beg] >= plat_thresh*fwc:
                i_beg -= 1
            while i_end < len(streak_row)-1 and streak_row[i_end] >= plat_thresh*fwc:
                i_end += 1

            i_begs = np.append(i_begs, i_beg)
            i_ends = np.append(i_ends, i_end)

    return i_begs, i_ends


def clean_row(streak_row, c_mask_row, t_mask_row, i_begs, i_ends, tail_filter):
    """Remove cosmic plateaus and tails from each streak row.

    Parameters
    ----------
    streak_row : array_like, float
        Row with cosmics.
    c_mask_row : array_like, int
        Row from cosmic mask.
    t_mask_row : array_like, int
        Row from tail mask.
    i_beg : array_like, int
        Indices of plateau beginnings.
    i_end : array_like, int
        Indices of plateau endings.
    tail_filter : int
        Moving median filter windows size for cosmic tail subtraction.

    Returns
    -------
    streak_row : array_like, float
        Row with cosmics removed and tails cleaned up
    """
    for i in range(len(i_begs)):
        i_beg = i_begs[i]
        i_end = i_ends[i]
        if i+1 < len(i_begs):
            tail_end = i_begs[i+1]  # Tail ends where next cosmic begins
        else:
            tail_end = len(streak_row) - 1  # Tail ends at end of row

        streak_row[i_beg:i_end+1] = 0.  # Remove cosmic hit
        # Subtract smoothed tail from cosmic tail
        smoothed_section = median_filter(streak_row[i_end+1:tail_end+1],
                                         tail_filter, mode='nearest')
        streak_row[i_end+1:tail_end+1] = (streak_row[i_end+1:tail_end+1]
                                          - smoothed_section)

        c_mask_row[i_beg:i_end+1] = 1
        t_mask_row[i_end+1:tail_end+1] = 1

    return streak_row, c_mask_row, t_mask_row
