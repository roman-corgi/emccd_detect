# -*- coding: utf-8 -*-
"""Example script for EMCCDDetect calls."""

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
    # Set up some inputs here
    here = os.path.abspath(os.path.dirname(__file__))
    # Get fluxmap
    fits_path = Path(here, 'data', 'sci_frame.fits')
    fluxmap = fits.getdata(fits_path).astype(float)  # (photons/pix/s)
    # Put fluxmap in 1024x1024 image section
    full_fluxmap = np.zeros((1024, 1024)).astype(float)
    full_fluxmap[0:fluxmap.shape[0], 0:fluxmap.shape[1]] = fluxmap
    # Specify frametime
    frametime = 100  # s


    # For the simplest possible use of EMCCDDetect, use its defaults
    emccd = EMCCDDetect()
    # Simulate only the fluxmap
    sim_sub_frame = emccd.sim_sub_frame(fluxmap, frametime)
    # Simulate the full frame (surround the full fluxmap with prescan, etc.)
    sim_full_frame = emccd.sim_full_frame(full_fluxmap, frametime)


    # For more control, each of the following parameters can be specified
    # Custom metadata path, if the user wants to use a different metadata file
    meta_path = Path(here, 'emccd_detect', 'util', 'metadata.yaml')
    # Note that the defaults for full_well_serial and eperdn are specified in
    # the metadata file
    emccd_spec = EMCCDDetect(
        meta_path=meta_path,
        em_gain=1.,
        full_well_image=60000.,  # e-
        full_well_serial=100000.,  # e-
        dark_current=0.0028,  # e-/pix/s
        cic=0.02,  # e-/pix/frame
        read_noise=100.,  # e-/pix/frame
        bias=10000.,  # e-
        qe=0.9,
        cr_rate=0.,  # hits/cm^2/s
        pixel_pitch=13e-6,  # m
        eperdn=7.,
        cic_gain_register=0.,  # e-/pix/frame
        numel_gain_register=604,
        nbits=14
    )


    # For legacy purposes, the class can also be called from a functon wrapper
    sim_old_style = emccd_detect(fluxmap, frametime, em_gain=5000.)


    # Plot images
    imagesc(full_fluxmap, 'Input Fluxmap')
    imagesc(sim_sub_frame, 'Output Sub Frame')
    imagesc(sim_full_frame, 'Output Full Frame')

    plt.show()
