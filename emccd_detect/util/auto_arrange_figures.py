# -*- coding: utf-8 -*-
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtWidgets

matplotlib.rcParams.update({'figure.max_open_warning': 0})


def auto_arrange_figures(nh=0, nw=0, monitor_id=0, rows_first=True):
    """Arrange all figures currently open.

    Call with no arguments to arrange automatically on primary monitor.

    Parameters
    ----------
    nh : int, optional
        Number of images in vertical direction. Defaults to 0.
    nw : int, optional
        Number of images in horizontal direction. Defaults to 0.
    monitor_id : int, optional
        Monitor number, where 0 is primary monitor. Defaults to 0.
    rows_first : bool, optional
        Option to organize from left to right rather than top to bottom.
        Defaults to True.

    Notes
    -----
    In automatic mode, if number of figures is greater than 40, figures will be
    stacked. In manual mode, if grid size is smaller than required for
    accommodating all figures, the function will switch to automatic mode.

    References
    ----------
    Adapted from Lee Jae Jun's Matlab function autoArrangeFigures [1]_.

    .. [1] leejaejun, Koreatech, Korea Republic, 2014.12.13, jaejun0201@gmail.com

    S Miller - UAH - 17-Dec-2018
    """
    task_bar_offset = [30, 50]

    if nh * nw == 0:
        auto_arange = True
    else:
        auto_arange = False

    fignums = plt.get_fignums()
    figs = np.array(list(map(plt.figure, plt.get_fignums())))

    fig_handle = _sort_figure_handles(fignums, figs)
    n_fig = fig_handle.shape[0]
    if n_fig <= 0:
        warnings.warn('Figures are not found.')
        return

    widget = QtWidgets.QDesktopWidget()

    screen = widget.screen(monitor_id)
    screen_sz = screen.geometry()
    scn_w = screen_sz.width() - task_bar_offset[0]
    scn_h = screen_sz.height() - task_bar_offset[1]
    scn_w_begin = screen_sz.left() + task_bar_offset[0]
    scn_h_begin = screen_sz.top() + task_bar_offset[1]

    if auto_arange:
        grid = np.array([[2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 4., 4.,
                          4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.,
                          4., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.],
                         [3., 3., 3., 3., 3., 3., 3., 3., 4., 4., 4., 5., 5.,
                          5., 5., 5., 5., 5., 5., 6., 6., 6., 7., 7., 7., 7.,
                          7., 7., 7., 7., 7., 7., 7., 7., 8., 8., 8., 8., 8.]])
        grid = grid.T

        if n_fig <= len(grid):
            nh = grid[n_fig - 1, 0]
            nw = grid[n_fig - 1, 1]
        # If more images than grid length, use largest grid and stack images.
        else:
            nh = grid[-1, 0]
            nw = grid[-1, 1]

        # If sceen is taller than it is wide, switch dimensions of image grid.
        if scn_h > scn_w:
            nh, nw = nw, nh

    fig_width = scn_w / nw
    fig_height = scn_h / nh

    for ipic in range(0, n_fig):
        if rows_first:
            [row, col] = _locate_r(ipic, nh, nw)
        else:
            [row, col] = _locate_c(ipic, nh, nw)

        window = fig_handle[ipic].canvas.manager.window
        window.setGeometry(scn_w_begin + fig_width*(col),
                           scn_h_begin + fig_height*(row),
                           fig_width,
                           fig_height)


def _locate_r(ipic, tot_rows, tot_cols):
    row = (int(np.floor((ipic)/tot_cols)) % tot_rows)
    col = (ipic % tot_cols)

    return [row, col]


def _locate_c(ipic, tot_rows, tot_cols):
    row = (ipic % tot_rows)
    col = (int(np.floor((ipic)/tot_rows)) % tot_cols)

    return [row, col]


def _sort_figure_handles(fignums, figs):
    idx = np.argsort(fignums)
    fig_sorted = figs[idx]

    return fig_sorted
