# -*- coding: utf-8 -*-
"""Example script for EMCCDDetect calls."""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from numpy.core.numeric import full_like

from emccd_detect.emccd_detect import EMCCDDetect, emccd_detect


def imagesc(data, title=None, vmin=None, vmax=None, cmap='viridis',
            aspect='equal', colorbar=True):
    """Plot a scaled colormap."""
    fig, ax = plt.subplots()
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect)

    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax)

    return fig, ax


if __name__ == '__main__':
    full_fluxmap = np.ones((1024, 1024))
    frametime = 100  # s
    em_gain = 100.

    emccd = EMCCDDetect(
        em_gain=em_gain,
        full_well_image=60000.,  # e-
        full_well_serial=100000.,  # e-
        dark_current=0.00,  # e-/pix/s
        cic=0.0,  # e-/pix/frame
        read_noise=0.,  # e-/pix/frame
        bias=10000.,  # e-
        qe=0.9,
        cr_rate=0.,  # hits/cm^2/s
        pixel_pitch=13e-6,  # m
        eperdn=10.,
        nbits=14,
        numel_gain_register=604
        )

    # Simulate the full frame (surround the full fluxmap with prescan, etc.)
    frames_l = []
    for i in range(2):
        sim_full_frame = emccd.sim_full_frame(full_fluxmap, frametime)
        frames_l.append(sim_full_frame)
    frames = np.stack(frames_l)

    # Plot images
    imagesc(frames[0], 'Output Full Frame')

    data = emccd.get_e_frame(emccd.slice_fluxmap(frames[0]).ravel())
    plt.figure()
    plt.hist(data, bins=50)
    plt.axvline(np.mean(data), c='r', ls='--')


    plt.show()
