# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


class ImagescException(Exception):
    """Exception class for imagesc module."""
    pass


def imagesc(data, title=None, vmin=None, vmax=None, cmap='viridis',
            aspect='equal', colorbar=True, grid=False):
    """Plot a scaled colormap.

    Parameters
    ----------
    data : array_like
        Input array.
    title : str, optional
        Plot title. Defaults to None.
    vmin : float, optional
        Minimum value to be mapped. Defaults to None.
    vmax : float, optional
        Maximum value to be mapped. Defaults to None.
    cmap : str, optional
        Matplotlib colormap. Defaults to viridis.
    aspect : str, optional
        Axes aspect ratio with respect to image. Defaults to equal.
    colorbar : bool, optional
        Option to display colorbar. Defaults to True.
    grid : bool, optional
        Option to display grid. Defaults to False.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Matplotlib Figure object.
    ax : Axes
        Matplotlib Axes object.

    Notes
    -----
    This is meant to loosely replicate Matlab's imagesc function.

    S Miller - UAH - 13-Feb-2019
    """
    fig, ax = plt.subplots()
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect)
    ax.grid(grid)

    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax)

    return fig, ax
