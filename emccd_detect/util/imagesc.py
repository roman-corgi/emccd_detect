# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


def imagesc(data, title=None, vmin=None, vmax=None, cmap='viridis',
            aspect='equal', colorbar=True, grid=False, extent=None):
    """Plot a scaled colormap.

    Parameters
    ----------
    data : array_like, shape (n, m)
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
    extent : scalars (left, right, bottom, top), optional
        

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
    if extent is None:
        im = ax.imshow(data, vmin=vmin, vmax=vmax, aspect=aspect, cmap=cmap)
    else:
        im = ax.imshow(data, vmin=vmin, vmax=vmax, aspect=aspect, cmap=cmap,
                       interpolation='none', extent=extent)
    ax.grid(grid)
    if title is not None:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax)

    return fig, ax
