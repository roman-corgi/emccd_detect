# -*- coding: utf-8 -*-
"""
Generate multiple frames and write to fits files in the proc_cgi_frame
directory.

"""
from __future__ import absolute_import, division, print_function

from os import path
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from emccd_detect.emccd_detect import EMCCDDetect


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
    # Input fluxmap
    fits_name = 'sci_frame.fits'
    current_path = Path(path.dirname(__file__))
    fits_path = Path(current_path, 'data', fits_name)
    fluxmap = fits.getdata(fits_path) * 2  # Input fluxmap (photons/pix/s)

    # Put fluxmap in 1024x1024 image section
    image = np.zeros((1024, 1024)).astype(float)
    image[0:fluxmap.shape[0], 0:fluxmap.shape[1]] = fluxmap

    proc_cgi_frame_path = '/Users/smiller/Documents/GitHub/proc_cgi_frame/'

    # Instantiate class
    emccd = EMCCDDetect(
        meta_path=Path(proc_cgi_frame_path, 'proc_cgi_frame', 'metadata.yaml'),
        em_gain=5000,
        full_well_image=60000,
        dark_current=0.0028,
        cic=0.01,
        read_noise=100,
        bias=0,
        qe=1,
        cr_rate=0,
        pixel_pitch=13e-6,
        shot_noise_on=True
    )

    frametime = 100  # s

    # Number of images to create
    nfiles = 5

    # Plot output images
    plot_images = True

    # Make brights
    path = '/Users/smiller/Documents/GitHub/proc_cgi_frame/data/sim/brights/'
    file_name = 'sim'
    for i in range(nfiles):
        sim_bright = emccd.sim_full_frame(image, frametime)
        fits.writeto(Path(path, '{}{}.fits'.format(file_name, i)),
                     sim_bright.astype(np.int32), overwrite=True)
        if plot_images:
            imagesc(sim_bright, 'Output Image')

    # Make darks
    path = '/Users/smiller/Documents/GitHub/proc_cgi_frame/data/sim/darks/'
    file_name = 'sim'
    for i in range(nfiles):
        sim_dark = emccd.sim_full_frame(np.zeros_like(image), frametime)
        fits.writeto(Path(path, '{}{}.fits'.format(file_name, i)),
                     sim_dark.astype(np.int32), overwrite=True)
        if plot_images:
            imagesc(sim_dark, 'Output Image')

    plt.show()
