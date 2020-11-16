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

    # Input fluxmap
    fits_name = 'sci_frame.fits'
    fits_path = Path(here, 'data', fits_name)
    fluxmap = fits.getdata(fits_path).astype(float)  # Input fluxmap (photons/pix/s)

    # Simulation inputs
    frametime = 100.  # Frame time (s)
    em_gain = 5000.  # CCD EM gain (e-/photon)
    full_well_image = 50000.  # Image area full well capacity (e-)
    full_well_serial = 90000.  # Serial (gain) register full well capacity (e-)
    dark_current = 0.0028  # Dark current rate (e-/pix/s)
    cic = 0.01  # Clock induced charge (e-/pix/frame)
    read_noise = 100.  # Read noise (e-/pix/frame)
    bias = 0.  # Bias offset (e-)
    qe = 0.9  # Quantum efficiency
    cr_rate = 0.  # Cosmic ray rate (5 for L2) (hits/cm^2/s)
    pixel_pitch = 13e-6  # Distance between pixel centers (m)
    shot_noise_on = True  # Apply shot noise

    # Traps at Hubble launch date
    date = 2452334.5
    traps, ccd, roe = model_for_HST_ACS(date)

    # Instantiate class
    emccd = EMCCDDetect(
        meta_path=Path(here, 'data', 'metadata.yaml'),
        em_gain=em_gain,
        full_well_image=full_well_image,
        full_well_serial=full_well_serial,
        dark_current=dark_current,
        cic=cic,
        read_noise=read_noise,
        bias=bias,
        qe=qe,
        cr_rate=cr_rate,
        pixel_pitch=pixel_pitch,
        shot_noise_on=shot_noise_on
    )

    # No traps
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

        # Plot gain divided arrays
        notrap = sim_frame_notrap/emccd.em_gain
        trap = sim_frame_trap/emccd.em_gain

        subtitle = f'Frametime = {int(frametime)}s'
        imagesc(expected_rate, 'Mean Expected Rate (e-/pix)\n' + subtitle)
        imagesc(notrap, 'Output Without Traps (e-/pix)\n' + subtitle,
                vmin=vmin, vmax=vmax)
        imagesc(trap, 'Output With Traps (e-/pix)\n' + subtitle,
                vmin=vmin, vmax=vmax)

        plt.show()
