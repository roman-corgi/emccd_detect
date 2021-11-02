"""Wrapper for read_metadata to allow use in emccd_detect simulator."""

import numpy as np

from emccd_detect.util.read_metadata import Metadata


class ReadMetadataWrapperException(Exception):
    """Exception class for read_metadata_wrapper module."""


class MetadataWrapper(Metadata):
    """Wrapper for Metadata class to add functionality for emccd_detect.

    Parameters
    ----------
    meta_path : str
        Full path of metadta yaml.

    Attributes
    ----------
    data : dict
        Data from metadata file.
    geom : SimpleNamespace
        Geometry specific data.
    eperdn : float
        Electrons per dn conversion factor (detector k gain).
    fwc : float
        Full well capacity of detector.
    sat_thresh : float
        Multiplication factor for fwc that determines saturated cosmic pixels.
    plat_thresh : float
        Multiplication factor for fwc that determines edges of cosmic plateu.
    cosm_filter : int
        Minimum length in pixels of cosmic plateus to be identified.
    tail_filter : int
        Moving median filter window size for cosmic tail subtraction.
    cic_thresh : float
        Multiplication factor for readnoise that determines beginning of cic.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get imaging area geometry
        self.rows_im, self.cols_im, self.ul_im = self._imaging_area_geom()

        # Make some zeros frames for initial creation of arrays
        self.imaging_area_zeros = np.zeros((self.rows_im, self.cols_im))
        self.full_frame_zeros = np.zeros((self.frame_rows, self.frame_cols))

    def mask(self, key):
        full_frame_m = self.full_frame_zeros.copy()

        rows, cols, ul = self._unpack_geom(key)
        full_frame_m[ul[0]:ul[0]+rows, ul[1]:ul[1]+cols] = 1
        return full_frame_m.astype(bool)

    def embed(self, frame, key, data):
        rows, cols, ul = self._unpack_geom(key)
        try:
            frame[ul[0]:ul[0]+rows, ul[1]:ul[1]+cols] = data
        except Exception:
            raise ReadMetadataWrapperException('Data does not fit in selected '
                                               'section')
        return frame

    def embed_im(self, im_area, key, data):
        rows, cols, ul = self._unpack_geom_im(key)
        try:
            im_area[ul[0]:ul[0]+rows, ul[1]:ul[1]+cols] = data
        except Exception:
            raise ReadMetadataWrapperException('Data does not fit in selected '
                                               'section')
        return im_area

    def imaging_slice(self, frame):
        """Select only the real counts from full frame and exclude virtual.

        Use this to transform mask and embed from acting on the full frame to
        acting on only the image frame.

        """
        rows, cols, ul = self._imaging_area_geom()

        return frame[ul[0]:ul[0]+rows, ul[1]:ul[1]+cols]

    def imaging_embed(self, frame, im_area):
        """Add the imaging area back to the full frame."""
        rows, cols, ul = self._imaging_area_geom()

        frame[ul[0]:ul[0]+rows, ul[1]:ul[1]+cols] = im_area
        return frame

    def _unpack_geom_corners(self, key):
        """Returns corners corresponding to geometry."""
        rows, cols, ul = self._unpack_geom(key)
        lr = (ul[0]+rows-1, ul[1]+cols-1)

        return ul, lr

    def _unpack_geom_im(self, key):
        """Wrapper for _unpack_geom, transforms ul locations from full frame
        coords to imaging area coords.

        """
        # Unpack geomotry of requested section
        rows, cols, ul_original = self._unpack_geom(key)
        # Unpack geometry of imaging area
        _, _, ul_im = self._imaging_area_geom()

        # Shift upper left locations by the upper left of the imaging area
        ul = ul_original.copy()
        ul[0] -= ul_im[0]
        ul[1] -= ul_im[1]

        # Make sure new geom is valid
        pass

        return rows, cols, ul

    def _imaging_area_geom(self):
        """Return geometry of imaging area in reference to full frame."""
        _, cols_pre, ul_prescan = self._unpack_geom('prescan')
        _, cols_serial_ovr, _ = self._unpack_geom('serial_overscan')
        rows_ovr, _, _ = self._unpack_geom('overscan')
        _, _, ul_image = self._unpack_geom('image')

        rows_im = self.frame_rows - rows_ovr
        cols_im = self.frame_cols - cols_pre - cols_serial_ovr
        ul_im = ul_image.copy()

        return rows_im, cols_im, ul_im

    def slice_section_im(self, im_area, key):
        """Slice 2d section out of imaging area of frame.

        Parameters
        ----------
        im_area : array_like
            Imaging area of frame, i.e. the full frame with the prescan and
            overscan removed:
                full_frame[:overscan[ul[0]], overscan[ul[1]]:]
        key : str
            Keyword referencing section to be sliced; must exist in geom and
            must not be 'prescan' or 'overscan'.

        """
        rows, cols, ul = self._unpack_geom_im(key)

        section = im_area[ul[0]:ul[0]+rows, ul[1]:ul[1]+cols]
        if section.size == 0:
            raise ReadMetadataWrapperException('Corners invalid')
        return section


if __name__ == '__main__':
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator


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


    here = os.path.abspath(os.path.dirname(__file__))
    meta_path = Path(here, 'metadata.yaml')
    meta = MetadataWrapper(meta_path)

    # Get masks of all regions
    image_m = meta.mask('image')
    prescan_m = meta.mask('prescan')
    overscan_m = meta.mask('overscan')
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
        + overscan_m*values['parallel_overscan']
        + serial_overscan_m*values['serial_overscan']
    )

    inverted = True
    plot_corners = True

    # Plot
    fig, ax = plt.subplots()
    ax.set_title('Science Frame Geometry')
    if inverted:
        im = ax.imshow(np.flipud(mask), origin='lower')
        if plot_corners:
            image_ul, image_lr = meta._unpack_geom_corners('image')
            plot_corner(ax, image_ul[1], meta.frame_rows-1 - image_ul[0], 'left', 'top', (5, -5))
            plot_corner(ax, image_ul[1], meta.frame_rows-1 - image_lr[0], 'left', 'bottom', (5, 5))
            plot_corner(ax, image_lr[1], meta.frame_rows-1 - image_lr[0], 'right', 'bottom', (-5, 5))
            plot_corner(ax, image_lr[1], meta.frame_rows-1 - image_ul[0], 'right', 'top', (-5, -5))
    else:
        im = ax.imshow(mask)
        if plot_corners:
            image_ul, image_lr = meta._unpack_geom_corners('image')
            plot_corner(ax, image_ul[1], image_ul[0], 'left', 'top', (5, -5))
            plot_corner(ax, image_ul[1], image_lr[0], 'left', 'bottom', (5, 5))
            plot_corner(ax, image_lr[1], image_lr[0], 'right', 'bottom', (-5, 5))
            plot_corner(ax, image_lr[1], image_ul[0], 'right', 'top', (-5, -5))

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
