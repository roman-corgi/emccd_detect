# -*- coding: utf-8 -*-
"""Example photon counting script."""
import numpy as np
import matplotlib.pyplot as plt

from emccd_detect.emccd_detect import EMCCDDetect
from photon_count.corr_photon_count import get_count_rate


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
    # Specify relevant detector properties (leave the other inputs at their
    # defaults)
    emccd = EMCCDDetect(
        em_gain=5000.,
        full_well_image=60000.,  # e-
        full_well_serial=100000.,  # e-
        dark_current=0.0028,  # e-/pix/s
        cic=0.02,  # e-/pix/frame
        read_noise=100.,  # e-/pix/frame
        bias=10000.,  # e-
        qe=0.9,
        cr_rate=0.,  # hits/cm^2/s
        eperdn=7.
    )

    # Set up incoming fluxmap
    rate = 0.1  # phot/pix/s
    fluxmap = np.ones((1024, 1024)) * rate  # Flat field

    # Simulate frames
    # Set frametime to get an output of 1 phot/pix
    frametime = 1/rate  # s
    frame_e_list = []
    frame_e_dark_list = []
    for i in range(10):
        # Simulate bright
        frame_dn = emccd.sim_sub_frame(fluxmap, frametime)
        # Simulate dark
        frame_dn_dark = emccd.sim_sub_frame(np.zeros_like(fluxmap), frametime)

        # Convert from dn to e- and bias subtract
        frame_e = frame_dn * emccd.eperdn - emccd.bias
        frame_e_dark = frame_dn_dark * emccd.eperdn - emccd.bias

        frame_e_list.append(frame_e)
        frame_e_dark_list.append(frame_e_dark)

    frame_e_cube = np.stack(frame_e_list)
    frame_e_dark_cube = np.stack(frame_e_dark_list)

    # Photon count, co-add, and correct for photometric error
    thresh = 5000.  # Use a high threshold to avoid undercount
    mean_rate = get_count_rate(frame_e_cube, thresh, emccd.em_gain)
    mean_rate_dark = get_count_rate(frame_e_dark_cube, thresh, emccd.em_gain)

    # Dark subtract
    mean_rate_ds = mean_rate - mean_rate_dark

    # Plot images
    imagesc(mean_rate_ds, f'Output Sub Frame\nMean: {np.mean(mean_rate_ds)}')
    plt.show()
