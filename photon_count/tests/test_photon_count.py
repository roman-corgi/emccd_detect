# -*- coding: utf-8 -*-
"""Unit tests for read_metadata."""
from __future__ import absolute_import, division, print_function

import unittest

import numpy as np

from proc_cgi_frame.photon_count import photon_count
from proc_cgi_frame.photon_count import PhotonCountException


class TestPhotonCount(unittest.TestCase):
    """Unit tests for photon_count function."""

    def setUp(self):
        self.thresh = 0.

    def test_empty_array(self):
        """Verify that function works for an empty array."""
        e_image = np.array([]).astype(float)
        pc_image = photon_count(e_image, self.thresh)
        self.assertTrue((pc_image == e_image).all())

    def test_single_array(self):
        """Verify that function works for an array with only one element."""
        e_image = np.ones(1)
        pc_image = photon_count(e_image, self.thresh)
        self.assertTrue((pc_image == e_image).all())

    def test_long_array(self):
        """Verify that function works for a very long array."""
        e_image = np.ones(1000000)
        pc_image = photon_count(e_image, self.thresh)
        self.assertTrue((pc_image == e_image).all())

    def test_two_dimensional_array(self):
        """Verify that function works for a two dimensional array."""
        e_image = np.ones([10, 10])
        pc_image = photon_count(e_image, self.thresh)
        self.assertTrue((pc_image == e_image).all())

    def test_thresh(self):
        """Verify that function correctly trims values."""
        e_image = np.array([1, 2, 3, 4, 5])
        thresh = 2.
        pc_image = photon_count(e_image, thresh)
        expected = np.array([0, 0, 1, 1, 1]).astype(float)
        self.assertTrue((pc_image == expected).all())

    def test_exception_not_array(self):
        """Verify that exception is thrown if input is not an array."""
        e_image = 1
        with self.assertRaises(PhotonCountException):
            photon_count(e_image, self.thresh)


if __name__ == '__main__':
    unittest.main()
