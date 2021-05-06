# -*- coding: utf-8 -*-
"""Comparison unit tests for rand_em_gain."""
from __future__ import absolute_import, division, print_function

import unittest
from os import path
from pathlib import Path

import matlab.engine
import numpy as np

from emccd_detect.rand_em_gain import rand_em_gain


# Current directory
here = Path(path.dirname(__file__))

# Start matlab and add emccd_detect_m directory
eng = matlab.engine.start_matlab()
eng.addpath(str(Path(here.parent, 'emccd_detect_m')), nargout=0)


def match_seeds():
    """Initialize numpy and Matlab random number generators to the same seed."""
    np.random.seed(1)
    eng.rng(1, 'twister', nargout=0)


class TestRandEmGain(unittest.TestCase):
    """Comparison unit tests for rand_em_gain function."""

    def setUp(self):
        match_seeds()
        self.em_gain = 1000.

    def test_compare_n0_array(self):
        """Verify that functions return the same value for n_in = 0."""
        n_in_array = np.zeros(100)
        out_py = rand_em_gain(n_in_array, self.em_gain)
        out_mat = eng.rand_em_gain(matlab.double(n_in_array.tolist()),
                                   self.em_gain)
        self.assertEqual(out_py.tolist(), np.array(out_mat)[0].tolist())

    def test_compare_n1_array(self):
        """Verify that functions return the same value for n_in = 1."""
        n_in_array = np.ones(100)
        out_py = rand_em_gain(n_in_array, self.em_gain)
        out_mat = eng.rand_em_gain(matlab.double(n_in_array.tolist()),
                                   self.em_gain)
        self.assertEqual(out_py.tolist(), np.array(out_mat)[0].tolist())

    def test_compare_n2_array(self):
        """Verify that functions return the same value for n_in = 2."""
        n_in_array = np.ones(100) * 2
        out_py = rand_em_gain(n_in_array, self.em_gain)
        out_mat = eng.rand_em_gain(matlab.double(n_in_array.tolist()),
                                   self.em_gain)
        self.assertEqual(out_py.tolist(), np.array(out_mat)[0].tolist())

    def test_compare_n3_array(self):
        """Verify that functions return the same value for n_in = 3."""
        n_in_array = np.ones(100) * 2
        out_py = rand_em_gain(n_in_array, self.em_gain)
        out_mat = eng.rand_em_gain(matlab.double(n_in_array.tolist()),
                                   self.em_gain)
        self.assertEqual(out_py.tolist(), np.array(out_mat)[0].tolist())


if __name__ == '__main__':
    unittest.main()
