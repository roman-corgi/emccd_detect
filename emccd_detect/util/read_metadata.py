# -*- coding: utf-8 -*-
"""Detector geometry metadata."""
from __future__ import absolute_import, division, print_function

import yaml


class ReadMetadataException(Exception):
    """Exception class for read_metadata module."""


class Metadata(object):
    """Create masks for different regions of the WFIRST CGI EMCCD detector.

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

    def __init__(self, meta_path):
        self.meta_path = meta_path

        self.data = self.get_data()
        self.frame_rows = self.data['frame_rows']
        self.frame_cols = self.data['frame_cols']
        self.geom = self.data['geom']
        self.eperdn = self.data['eperdn']
        self.fwc = self.data['fwc']
        self.sat_thresh = self.data['sat_thresh']
        self.plat_thresh = self.data['plat_thresh']
        self.cosm_filter = self.data['cosm_filter']
        self.tail_filter = self.data['tail_filter']
        self.cic_thresh = self.data['cic_thresh']

    def get_data(self):
        """Read yaml data into dictionary."""
        with open(self.meta_path, 'r') as stream:
            data = yaml.safe_load(stream)
        return data

    def slice_section(self, frame, key):
        """Slice 2d section out of frame.

        Parameters
        ----------
        frame : array_like
            Full frame consistent with size given in frame_rows, frame_cols.
        key : str
            Keyword referencing section to be sliced; must exist in geom.

        """
        rows, cols, ul = self._unpack_geom(key)

        section = frame[ul[0]:ul[0]+rows, ul[1]:ul[1]+cols]
        if section.size == 0:
            raise ReadMetadataException('Corners invalid')
        return section

    def _unpack_geom(self, key):
        """Safely check format of geom sub-dictionary and return values."""
        # XXX Need to do key checking later; for now just unpack
        coords = self.geom[key]
        rows = coords['rows']
        cols = coords['cols']
        ul = coords['ul']

        return rows, cols, ul
