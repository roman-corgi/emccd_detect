# -*- coding: utf-8 -*-
"""Script to show the effect of traps on detector output."""
from __future__ import absolute_import, division, print_function

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from emccd_detect.emccd_detect import EMCCDDetect
from arcticpy.main import model_for_HST_ACS


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
    meta_path = Path(here, 'data', 'metadata.yaml')

    # Set up input fluxmap
    fits_path = Path(here, 'data', 'sci_frame.fits')
    fluxmap = fits.getdata(fits_path).astype(float)  # (photons/pix/s)

    # Traps at Hubble launch date
    date = 2452334.5
    traps, ccd, roe = model_for_HST_ACS(date)

    # Instantiate class
    emccd = EMCCDDetect(
        meta_path=meta_path,
        em_gain=5000.,
        full_well_image=60000,  # e-
        dark_current=0.0028,  # e-/pix/s
        cic=0.02,  # e-/pix/s
        read_noise=100,  # e-
        bias=10000,  # e-
        qe=0.9,
        cr_rate=0.,  # hits/cm^2/s
        pixel_pitch=13e-6,  # m
        shot_noise_on=True,
        cic_gain_register=0.,  # e-/pix/s
        numel_gain_register=604,
        nbits=14
    )

    # No traps
    frametime = 100  # s
    sim_frame_notrap = emccd.sim_sub_frame(fluxmap, frametime)

    # Expected rate for given run
    expected_rate = emccd.mean_expected_rate

    # Traps
    emccd.update_cti(ccd=ccd, roe=roe, traps=traps, express=1)
    sim_frame_trap = emccd.sim_sub_frame(fluxmap, frametime)

    # Plot images
    plot_images = True
    if plot_images:
        imagesc(fluxmap, 'Input Fluxmap (phot/pix/s)')

        vmin = np.min(expected_rate)
        vmax = np.max(expected_rate)

        # Convert output arrays from dn to electrons
        notrap = (sim_frame_notrap*emccd.eperdn - emccd.bias) / emccd.em_gain
        trap = (sim_frame_trap*emccd.eperdn - emccd.bias) / emccd.em_gain

        subtitle = f'Frametime = {int(frametime)}s'
        imagesc(expected_rate, 'Mean Expected Rate (e-/pix)\n' + subtitle)
        imagesc(notrap, 'Output Without Traps (e-/pix)\n' + subtitle,
                vmin=vmin, vmax=vmax)
        imagesc(trap, 'Output With Traps (e-/pix)\n' + subtitle,
                vmin=vmin, vmax=vmax)

        plt.show()
