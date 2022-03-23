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
        self.rows_im, self.cols_im, self.r0c0_im = self._imaging_area_geom()

        # Make some zeros frames for initial creation of arrays
        self.imaging_area_zeros = np.zeros((self.rows_im, self.cols_im))
        self.full_frame_zeros = np.zeros((self.frame_rows, self.frame_cols))

    def mask(self, key):
        full_frame_m = self.full_frame_zeros.copy()

        rows, cols, r0c0 = self._unpack_geom(key)
        full_frame_m[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = 1
        return full_frame_m.astype(bool)

    def embed(self, frame, key, data):
        rows, cols, r0c0 = self._unpack_geom(key)
        try:
            frame[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = data
        except Exception:
            raise ReadMetadataWrapperException('Data does not fit in selected '
                                               'section')
        return frame

    def embed_im(self, im_area, key, data):
        rows, cols, r0c0 = self._unpack_geom_im(key)
        try:
            im_area[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = data
        except Exception:
            raise ReadMetadataWrapperException('Data does not fit in selected '
                                               'section')
        return im_area

    def imaging_slice(self, frame):
        """Select only the real counts from full frame and exclude virtual.

        Use this to transform mask and embed from acting on the full frame to
        acting on only the image frame.

        """
        rows, cols, r0c0 = self._imaging_area_geom()

        return frame[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols]

    def imaging_embed(self, frame, im_area):
        """Add the imaging area back to the full frame."""
        rows, cols, r0c0 = self._imaging_area_geom()

        frame[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = im_area
        return frame

    def _unpack_geom_corners(self, key):
        """Returns corners corresponding to geometry."""
        rows, cols, r0c0 = self._unpack_geom(key)
        r1c1 = (r0c0[0]+rows-1, r0c0[1]+cols-1)

        return r0c0, r1c1

    def _unpack_geom_im(self, key):
        """Wrapper for _unpack_geom, transforms r0c0 locations from full frame
        coords to imaging area coords.

        """
        # Unpack geomotry of requested section
        rows, cols, r0c0_original = self._unpack_geom(key)
        # Unpack geometry of imaging area
        _, _, r0c0_im = self._imaging_area_geom()

        # Shift r0c0 locations by the r0c0 of the imaging area
        r0c0 = r0c0_original.copy()
        r0c0[0] -= r0c0_im[0]
        r0c0[1] -= r0c0_im[1]

        # Make sure new geom is valid
        pass

        return rows, cols, r0c0

    def _imaging_area_geom(self):
        """Return geometry of imaging area in reference to full frame."""
        _, cols_pre, _ = self._unpack_geom('prescan')
        _, cols_serial_ovr, _ = self._unpack_geom('serial_overscan')
        rows_parallel_ovr, _, _ = self._unpack_geom('parallel_overscan')
        #_, _, r0c0_image = self._unpack_geom('image')
        fluxmap_rows, _, r0c0_image = self._unpack_geom('image')

        rows_im = self.frame_rows - rows_parallel_ovr
        cols_im = self.frame_cols - cols_pre - cols_serial_ovr
        r0c0_im = r0c0_image.copy()
        r0c0_im[0] = r0c0_im[0] - (rows_im - fluxmap_rows)

        return rows_im, cols_im, r0c0_im

    def slice_section_im(self, im_area, key):
        """Slice 2d section out of imaging area of frame.

        Parameters
        ----------
        im_area : array_like
            Imaging area of frame, i.e. the full frame with the prescan and
            overscan removed:
                full_frame[:overscan[r0c0[0]], overscan[r0c0[1]]:]
        key : str
            Keyword referencing section to be sliced; must exist in geom and
            must not be 'prescan' or 'overscan'.

        """
        rows, cols, r0c0 = self._unpack_geom_im(key)

        section = im_area[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols]
        if section.size == 0:
            raise ReadMetadataWrapperException('Corners invalid')
        return section
