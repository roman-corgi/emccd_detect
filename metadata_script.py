import numpy as np
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

from emccd_detect.emccd_detect import MetadataWrapper


def plot_corner(ax, x, y, ha, va, xytext):
    """Plot marker with coordinates label."""
    ax.scatter(x, y, s=2, c='r', marker='s')
    ax.annotate(f'({x}, {y})', (x, y), size=7, ha=ha, va=va, xytext=xytext,
                textcoords='offset pixels')


class Formatter(object):
    """Round cursor coordinates to integers."""
    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        return 'x=%i, y=%i' % (np.round(x), np.round(y))


if __name__ == '__main__':
    here = os.path.abspath(os.path.dirname(__file__))
    meta_path = Path(here, 'emccd_detect', 'util', 'metadata.yaml')
    meta = MetadataWrapper(meta_path)

    # Get masks of all regions
    image_m = meta.mask('image')
    prescan_m = meta.mask('prescan')
    parallel_overscan_m = meta.mask('parallel_overscan')
    serial_overscan_m = meta.mask('serial_overscan')

    # Assign values to each region
    values = {
        'image': 1,
        'prescan': 0.75,
        'parallel_overscan': 0.5,
        'serial_overscan': 0.25,
        'shielded': 0.
    }

    # Stack masks
    mask = (
        image_m*values['image']
        + prescan_m*values['prescan']
        + parallel_overscan_m*values['parallel_overscan']
        + serial_overscan_m*values['serial_overscan']
    )

    # Plot
    fig, ax = plt.subplots()
    ax.set_title('Science Frame Geometry')
    im = ax.imshow(mask, origin='lower')
    # Plot corners
    image_r0c0, image_r1c1 = meta._unpack_geom_corners('image')
    plot_corner(ax, image_r0c0[1], image_r0c0[0], 'left', 'bottom', (5, 5))
    plot_corner(ax, image_r0c0[1], image_r1c1[0], 'left', 'top', (5, -5))
    plot_corner(ax, image_r1c1[1], image_r1c1[0], 'right', 'top', (-5, -5))
    plot_corner(ax, image_r1c1[1], image_r0c0[0], 'right', 'bottom', (-5, 5))

    # Format plot
    ax.format_coord = Formatter(im)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Make legend
    colors = {region: im.cmap(im.norm(value))
              for region, value in values.items()}
    patches = {mpatches.Patch(color=color, label=f'{region}')
               for region, color in colors.items()}
    plt.legend(handles=patches, loc='lower left')

    plt.show()
