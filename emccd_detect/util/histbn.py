# -*- coding: utf-8 -*-
import copy

import matplotlib.pyplot as plt
import numpy as np


def histbn(x, bounds='auto', nbins='auto', bin_width='auto',
           graph_outliers=False, log=False, suppress_warning=False):
    """Plot a bounded histogram and display underflow and overflow values.

    Parameters
    ----------
    x : array_like
        Input data.
    bounds : :obj:`list` of :obj:`float` or :obj:`str`, optional
        Bounds of histogram. For default +/- 5 sigma bounds, set to 'auto'. For
        no bounds, set bounds to 'all'. Defaults to 'auto'.
    nbins : :obj:`int` or :obj:`str`, optional
        Number of bins. For default sqrt(x.size) number of bins, set 'auto'.
        Defaults to 'auto'.
    bin_width : :obj:`float` or :obj:`str`, optional
        Bin width. Note that manually setting bin_width will override nbins.
        For default bin width that is based on number of bins, set to 'auto'.
        Defaults to 'auto'.
    graph_outliers : bool, optional
        Option to reate additional histograms for underflow and overflow data,
        if any. Defaults to False.
    log : bool, optional
        Option to plot histogram on a logarithmic y axis. Defaults to False.
    suppress_warning : bool, optional
        Option to suppress the warning symbol (> !) as well as output note
        indicating bin bounds being stretched to match bin width. Defaults to
        False.

    Returns
    -------
    out : instance
        Class instance containing histbn properties. Attributes are as follows:

        data : :obj:`ndarray` of :obj:`float`
            Output data. This is the bounded data that is plotted to the
            histogram.
        values : :obj:`ndarray` of :obj:`float`
            Bounded histogram values.
        nbins : int
            Number of bins for bounded histogram.
        bin_edges : :obj:`ndarray` of :obj:`float`
            Bin edges for bounded histogram.
        bin_width : float
            Bin width for bounded histogram.
        bin_limits : :obj:`list` of :obj:`float`
            Bounds of histogram.
        handle : :obj:`tuple` of :obj:`ndarray` of :obj:`float`
            Output of ax.hist.
        fig : :class:`matplotlib.figure.Figure`
            Matplotlib figure object.
        ax : Axes
            Matplotlib axes object
        mean : float
            Mean of bounded histogram data.
        sdev : float
            Standard deviation of bounded histogram data.
        data_total : int
            Total number of input data points.
        data_cut : int
            Number of data points plotted in bounded histogram.
        underflow_data : :obj:`ndarray` of :obj:`float`
            Data below histogram bounds.
        overflow_data : :obj:`ndarray` of :obj:`float`
            Data above histogram bounds.
        underflow : int
            Number of underflow data points.
        overflow : int
            Number of overflow data points.

    Notes
    -----
    Note that the histogram properties (number of entries, standard deviation,
    and mean) refer to the bounded data and not the original dataset.

    References
    ----------
    Adapted from histoutline by Matt Foster. ee1mpf@bath.ac.uk

    B Nemati and S Miller - UAH - 10-Jul-2018
    """
    x = np.array(x)
    if x.ndim != 1:
        x = x.reshape(x.size)

    if nbins == 'auto':
        nbins = int(round(np.sqrt(len(x))))

    if bounds == 'all':
        bounds = [min(x), max(x)]

    x_len = len(x)
    x_mean = np.mean(x)
    x_std = np.std(x, ddof=1)

    if not bounds or bounds == 'auto':
        thresh = 5
        bounds = [x_mean - thresh*x_std, x_mean + thresh*x_std]

    underflow = x[x < bounds[0]]
    overflow = x[x > bounds[1]]
    # cut is the data that will be plotted
    cut = x[np.in1d(x, np.concatenate((underflow, overflow)), invert=True)]
    cut_len = len(cut)
    cut_mean = np.mean(cut)
    cut_std = np.std(cut, ddof=1)
    underflow_len = len(underflow)
    overflow_len = len(overflow)

    fig, ax = plt.subplots()

    note_str = False
    if bin_width == 'auto':
        h = ax.hist(cut, bins=nbins, histtype='step', range=bounds)
        bin_width = h[1][1] - h[1][0]
    else:
        over_bound = np.mod(bounds[0]-bounds[1], bin_width)
        if over_bound:
            ibound = copy.copy(bounds[1])
            bounds[1] = bounds[1] + over_bound
            if not suppress_warning:
                ax.annotate('> !', xy=(0.95, 0.02), xycoords=('axes fraction'))
                note_str = '>! Right bound stretched from {:.3f} to {:.3f} ' \
                           'to match bin width.'.format(ibound, bounds[1])
        bin_width_edges = np.arange(min(cut), max(cut)+bin_width, bin_width)
        h = ax.hist(cut, bins=bin_width_edges, histtype='step', range=bounds)

    if log:
        ax.set_yscale('log')
        ax.set_ylim(0.1)

    ax.set_xlim(bounds[0], bounds[1])

    tick_width = ax.get_yticks()[-1]-ax.get_yticks()[-2]
    if ax.get_ylim()[1]-tick_width <= max(h[0]):
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + tick_width)

    str1 = 'entries = {:}'.format(cut_len)
    str2 = '\nmean   = {:.2e}'.format(cut_mean)
    str3 = '\nstdev    = {:.2e}'.format(cut_std)
    str4 = '\nunderflow = {:}'.format(underflow_len)
    str5 = '\noverflow   = {:}'.format(overflow_len)
    string = str1 + str2 + str3 + str4 + str5

    bbox = dict(boxstyle='round', facecolor='white', alpha=0.8,
                edgecolor='0.7')
    ax.annotate(string, (0.6, 0.5), size=10, xycoords='figure fraction',
                bbox=bbox)

    ax.set_ylabel('Entries / Bin  (size {:.3f})'.format(bin_width))

    class HistBN:
        pass
    out = HistBN()

    if graph_outliers:
        if underflow_len != 0:
            fig1, ax1 = plt.subplots()
            bin_width_edges_under = np.arange(min(underflow),
                                              max(underflow)+bin_width,
                                              bin_width)
            h_under = ax1.hist(underflow, bin_width_edges_under,
                               histtype='step')
            ax1.set_title('Underflow')
            ax1.set_ylabel('Entries / Bin  (size {:.3f})'.format(bin_width))

            if log:
                ax1.set_yscale('log')
                ax1.set_ylim(0.1)
            ax1.set_xlim(ax1.get_xlim()[0], h_under[1][-1])

            out.h_under = h_under
            out.figunder = fig1
            out.axunder = ax1

        if overflow_len != 0:
            fig2, ax2 = plt.subplots()
            bin_width_edges_over = np.arange(min(overflow),
                                             max(overflow)+bin_width,
                                             bin_width)
            h_over = ax2.hist(overflow, bin_width_edges_over,
                              histtype='step')
            ax2.set_title('Overflow')
            ax2.set_ylabel('Entries / Bin  (size {:.3f})'.format(bin_width))
            if log:
                ax2.set_yscale('log')
                ax2.set_ylim(0.1)
            ax2.set_xlim(h_over[1][0], ax2.get_xlim()[1])

            out.h_over = h_over
            out.figover = fig2
            out.axover = ax2

    fig.show()  # Return to viewing main histogram

    out.data = cut
    out.values = h[0]
    out.nbins = len(h[0])
    out.bin_edges = h[1]
    out.bin_width = h[1][1] - h[1][0]
    out.bin_limits = [h[1][0], h[1][-1]]
    out.handle = h
    out.fig = fig
    out.ax = ax

    out.mean = cut_mean
    out.sdev = cut_std
    out.data_total = x_len
    out.data_cut = cut_len
    out.underflow_data = underflow
    out.overflow_data = overflow
    out.underflow = underflow_len
    out.overflow = overflow_len
    if note_str:
        out.note = note_str

    return out
