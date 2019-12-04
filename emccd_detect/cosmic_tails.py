# -*- coding: utf-8 -*-
import numpy as np


def cosmic_tails(frame, matrixh, matrixw, fwc_gr, h, k, r):
    """Generate cosmic tails from cosmic hits.

    Parameters
    ----------
    frame : :obj:`ndarray` of :obj:`float`
        Input array.
    matrixh : int
        Height of array (pix).
    matrixw : int
        Width of array (pix).
    fwc_gr : float
        Full well capacity, gain register.
    h : :obj:`ndarray` of :obj:`float`
        Column coordinates of cosmic hits (pix).
    k : :obj:`ndarray` of :obj:`float`
        Row coordinates of cosmic hits (pix).
    r : :obj:`ndarray` of :obj:`float`
        Radii of cosmic hits (pix).

    Returns
    -------
    frame : :obj:`ndarray` of :obj:`float`
        Output array.

    S Miller - UAH - 16-Jan-2019
    """
    # tails are composed of one rapidly decaying exponential and another
    # gradually decaying, with a smooth transition between
    # These must be floats
    n1 = 0.010  # 0.010
    n2 = 0.250  # 0.250
    n3 = 1.000  # 1.000 controls weight between exponentials
    a = 0.970  # 0.970
    b = 0.030  # 0.030

    # create tails
    for j in np.arange(0, len(k)):
        rows = np.arange(max(k[j]-r[j], 0), min(k[j]+r[j], matrixh-1)+1)
        cols = np.arange(max(h[j]-r[j], 0), min(h[j]+r[j], matrixw-1)+1)
        nonzero = np.sum(frame[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1] > fwc_gr, 1)  # noqa: E501
        rowsi = rows[nonzero != 0]

        for ii in rowsi:
            overflow_sum = 0
            col_start = (frame[ii, cols[0]:cols[-1]+1] > fwc_gr).nonzero()[0][0]  # noqa: E501
            jj = cols[col_start]

            while frame[ii, jj] > fwc_gr:
                overflow = frame[ii, jj] - fwc_gr
                overflow_sum += overflow
                next_ = min(jj+1, matrixw-1)
                frame[ii, next_] = frame[ii, next_] + overflow
                frame[ii, jj] = fwc_gr
                jj = next_
                tail_start = jj-1

            scale = 20  # scale tails to match those in sample images
            taillen = int(round(overflow_sum/fwc_gr) * scale)
            tail_row_end = min(matrixw-1, tail_start + taillen-1)
            n = np.linspace(0, 1, taillen)
            taila = a*np.exp(-n/n1) * 1/np.exp(n/n3)
            tailb = b*np.exp(-n/n2) * 1/np.exp(-n/n3)
            tail = frame[ii, tail_start] * (taila + tailb)
            frame_slice = np.arange(tail_start, tail_row_end+1)
            tail_slice = np.arange(0, (tail_row_end+1)-tail_start)
            if frame_slice.size > 0:
                frame[ii, frame_slice[0]:frame_slice[-1]+1] += tail[tail_slice]

    return frame
