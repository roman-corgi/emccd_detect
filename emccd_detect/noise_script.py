# -*- coding: utf-8 -*-
"""Get histograms of noise."""
import numpy as np
import matplotlib.pyplot as plt

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
    #full_fluxmap = np.ones((1024, 1024))
    fluxmap = np.ones((100,100))
    frametime = 0.1  # s (adjust lambda by adjust this)
    em_gain = 5000.

    emccd = EMCCDDetect(
        em_gain=em_gain,
        full_well_image=60000.,  # e-
        full_well_serial=100000.,  # e-
        dark_current=0.00,  # e-/pix/s
        cic=0.0,  # e-/pix/frame
        read_noise=0.,  # e-/pix/frame
        bias=0, # 10000.,  # e-
        qe=1,  # set this to 1 so it doesn't affect lambda
        cr_rate=0.,  # hits/cm^2/s
        pixel_pitch=13e-6,  # m
        eperdn=1.,  # set this to 1 so there's no data loss when converting back to e-
        nbits=64,
        numel_gain_register=604
        )

    # Simulate several full frames
    frames_l = []
    nframes = 500
    for i in range(nframes):
        #sim_full_frame = emccd.sim_full_frame(full_fluxmap, frametime)
        sim_sub_frame = emccd.sim_sub_frame(fluxmap,frametime)
        e_frame = emccd.get_e_frame(sim_sub_frame)
        frames_l.append(e_frame)
    frames = np.stack(frames_l)

    # Plot images
    #imagesc(emccd.get_e_frame(frames[0]), 'Output Full Frame')

    f, ax = plt.subplots(1,2)
    ax[0].hist(np.mean(frames,axis=0).flatten(), bins=20)
    ax[0].axvline(np.mean(fluxmap)*frametime, color='black')
    ax[0].set_title('Pixel mean')
    ax[1].hist(np.std(frames,axis=0).flatten(), bins=20)
    ax[1].axvline(np.sqrt(np.mean(fluxmap)*frametime),color='black')
    ax[1].axvline(np.sqrt(2*np.mean(fluxmap)*frametime),color='red')
    ax[1].set_title('Pixel sdev')
    plt.tight_layout()
    plt.show()

    # data = emccd.get_e_frame(emccd.slice_fluxmap(frames[0]).ravel())
    # plt.figure()
    # plt.hist(data, bins=50)
    # plt.title(f'em gain = {em_gain}, lambda = {np.mean(full_fluxmap) * frametime}')
    # plt.xlabel('counts (e-)')

    # plt.show()
