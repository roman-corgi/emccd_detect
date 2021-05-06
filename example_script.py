# -*- coding: utf-8 -*-
"""Example script for EMCCDDetect calls."""
from __future__ import absolute_import, division, print_function

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

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
    here = os.path.abspath(os.path.dirname(__file__))

    # Specify metadata path
    meta_path = Path(here, 'emccd_detect', 'util', 'metadata.yaml')

    # Set up input fluxmap
    fits_path = Path(here, 'data', 'sci_frame.fits')
    fluxmap = fits.getdata(fits_path).astype(float)  # (photons/pix/s)
    # Put fluxmap in 1024x1024 image section
    full_fluxmap = np.zeros((1024, 1024)).astype(float)
    full_fluxmap[0:fluxmap.shape[0], 0:fluxmap.shape[1]] = fluxmap


    # Instantiate class
    # Note that full_well_serial and eperdn will default to the numbers in
    # metadata.yaml if not specified here
    emccd = EMCCDDetect(
        meta_path=meta_path,
        em_gain=1.,
        full_well_image=60000.,
        full_well_serial=100000.,
        dark_current=0.0028,
        cic=0.02,
        read_noise=100.,
        bias=10000.,
        qe=0.9,
        cr_rate=0.,
        pixel_pitch=13e-6,
        eperdn=7.,
        cic_gain_register=0.,
        numel_gain_register=604,
        nbits=14
    )

    # Simulate full frame
    frametime = 100
    sim_frame = emccd.sim_full_frame(full_fluxmap, frametime)

    # Alternatively, add noise just to the input fluxmap
    sim_sub_frame = emccd.sim_sub_frame(fluxmap, frametime)

    # For legacy purposes, the class can also be called from a functon wrapper
    sim_old_style = emccd_detect(fluxmap, frametime, em_gain=5000., shot_noise_on=True)

    # Plot images
    # Full frame images
    imagesc(full_fluxmap, 'Input Fluxmap')
    imagesc(sim_frame, 'Output Frame')

    plt.show()
