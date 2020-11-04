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

    B Nemati and S Miller - UAH - 03-Aug-2018

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_frame_zeros = np.zeros((self.frame_rows, self.frame_cols),
                                         dtype=int)

    def mask(self, key):
        full_frame_m = self.full_frame_zeros.copy()

        rows, cols, ul = self._unpack_geom(key)
        full_frame_m[ul[0]:ul[0]+rows, ul[1]:ul[1]+cols] = 1
        return full_frame_m.astype(bool)

    def embed(self, frame, key, data):
        frame = self.full_frame_zeros.copy()

        rows, cols, ul = self._unpack_geom(key)
        try:
            frame[ul[0]:ul[0]+rows, ul[1]:ul[1]+cols] = data
        except Exception:
            raise ReadMetadataWrapperException('Data does not fit in selected '
                                               'section')
        return frame

    def embed_im(self, frame_im, key, data):
        rows, cols, ul = self._unpack_geom_im(key)
        try:
            frame_im[ul[0]:ul[0]+rows, ul[1]:ul[1]+cols] = data
        except Exception:
            raise ReadMetadataWrapperException('Data does not fit in selected '
                                               'section')
        return frame_im

    def imaging_slice(self, frame):
        """Select only the real counts from full frame and exclude virtual.

        Use this to transform mask and embed from acting on the full frame to
        acting on only the image frame.

        """
        rows_pre, cols_pre, ul_pre = self._unpack_geom('prescan')
        rows_ovr, cols_ovr, ul_ovr = self._unpack_geom('overscan')
        return frame[0:ul_ovr[0], ul_pre[1]+cols_pre:]

    def imaging_embed(self, full_frame, frame_im):
        """Add the imaging area back to the full frame."""
        rows_pre, cols_pre, ul_pre = self._unpack_geom('prescan')
        rows_ovr, cols_ovr, ul_ovr = self._unpack_geom('overscan')

        full_frame[0:ul_ovr[0], ul_pre[1]+cols_pre:] = frame_im
        return full_frame

    def _unpack_geom_im(self, key):
        """Wrapper for _unpack_geom, transforms ul locations from full frame
        coords to imaging area coords.

        """
        rows, cols, ul_original = self._unpack_geom(key)

        # Shift all upper left locations by the upper left of dark_ref_top.
        _, _, ul_dark_ref_top = self._unpack_geom('dark_ref_top')
        ul = ul_original.copy()
        ul[1] -= ul_dark_ref_top[1]

        return rows, cols, ul

    def slice_section_im(self, frame_im, key):
        """Slice 2d section out of imaging area of frame.

        Parameters
        ----------
        frame_im : array_like
            Imaging area of frame, i.e. the full frame with the prescan and
            overscan removed:
                full_frame[:overscan[ul[0]], overscan[ul[1]]:]
        key : str
            Keyword referencing section to be sliced; must exist in geom and
            must not be 'prescan' or 'overscan'.

        """
        rows, cols, ul = self._unpack_geom_im(key)

        section = frame_im[ul[0]:ul[0]+rows, ul[1]:ul[1]+cols]
        if section.size == 0:
            raise ReadMetadataWrapperException('Corners invalid')
        return section
