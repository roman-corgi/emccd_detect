import numpy as np
import pytest

import arcticpy as ac


class TestCCD:
    def test__single_electron_fractional_height_from_electrons(
        self,
    ):

        parallel_ccd = ac.CCD(
            full_well_depth=10000.0, well_notch_depth=0.0, well_fill_power=1.0
        )

        electron_fractional_volumes = (
            parallel_ccd.cloud_fractional_volumes_from_n_electrons_and_phase(
                n_electrons=100.0
            )
        )

        assert electron_fractional_volumes == 0.01

        electron_fractional_volumes = (
            parallel_ccd.cloud_fractional_volumes_from_n_electrons_and_phase(
                n_electrons=1000.0
            )
        )

        assert electron_fractional_volumes == 0.1

        electron_fractional_volumes = (
            parallel_ccd.cloud_fractional_volumes_from_n_electrons_and_phase(
                n_electrons=1000000.0
            )
        )

        assert electron_fractional_volumes == 1.0

        parallel_ccd = ac.CCD(
            full_well_depth=10000.0, well_notch_depth=0.0, well_fill_power=0.5
        )

        electron_fractional_volumes = (
            parallel_ccd.cloud_fractional_volumes_from_n_electrons_and_phase(
                n_electrons=100.0
            )
        )

        assert electron_fractional_volumes == 0.01 ** 0.5

        electron_fractional_volumes = (
            parallel_ccd.cloud_fractional_volumes_from_n_electrons_and_phase(
                n_electrons=1000.0
            )
        )

        assert electron_fractional_volumes == 0.1 ** 0.5

        parallel_ccd = ac.CCD(
            full_well_depth=100.0, well_notch_depth=90.0, well_fill_power=1.0
        )

        electron_fractional_volumes = (
            parallel_ccd.cloud_fractional_volumes_from_n_electrons_and_phase(
                n_electrons=100.0
            )
        )

        assert electron_fractional_volumes == 1.0

        electron_fractional_volumes = (
            parallel_ccd.cloud_fractional_volumes_from_n_electrons_and_phase(
                n_electrons=9.0
            )
        )

        assert electron_fractional_volumes == 0.0

    def test__electron_fractional_heights_from_electrons(
        self,
    ):

        parallel_ccd = ac.CCD(
            full_well_depth=10000.0, well_notch_depth=0.0, well_fill_power=1.0
        )

        electron_fractional_volumes = (
            parallel_ccd.cloud_fractional_volumes_from_n_electrons_and_phase(
                n_electrons=[10.0, 100.0, 1000.0]
            )
        )

        assert electron_fractional_volumes == pytest.approx([0.001, 0.01, 0.1])


class TestMultiPhase:
    def test__mutli_phase_initialisation(self):
        # All duplicated
        ccd = ac.CCD(
            well_notch_depth=0.01,
            well_fill_power=0.8,
            full_well_depth=84700,
            fraction_of_traps_per_phase=[0.5, 0.2, 0.2, 0.1],
        )

        assert ccd.well_notch_depth == [0.01] * 4
        assert ccd.well_fill_power == [0.8] * 4
        assert ccd.full_well_depth == [84700] * 4

        # Some duplicated
        ccd = ac.CCD(
            well_notch_depth=0.01,
            well_fill_power=0.8,
            full_well_depth=[84700, 1e5, 2e5, 3e5],
            fraction_of_traps_per_phase=[0.5, 0.2, 0.2, 0.1],
        )

        assert ccd.well_notch_depth == [0.01] * 4
        assert ccd.well_fill_power == [0.8] * 4
        assert ccd.full_well_depth == [84700, 1e5, 2e5, 3e5]

    def test__extract_phase(self):
        ccd = ac.CCD(
            well_notch_depth=0.01,
            well_fill_power=0.8,
            full_well_depth=[84700, 1e5, 2e5, 3e5],
            fraction_of_traps_per_phase=[0.5, 0.2, 0.2, 0.1],
        )

        ccd_phase_0 = ac.CCDPhase(ccd, 0)
        ccd_phase_1 = ac.CCDPhase(ccd, 1)
        ccd_phase_2 = ac.CCDPhase(ccd, 2)
        ccd_phase_3 = ac.CCDPhase(ccd, 3)

        assert ccd_phase_0.well_notch_depth == 0.01
        assert ccd_phase_0.full_well_depth == 84700
        assert ccd_phase_1.well_notch_depth == 0.01
        assert ccd_phase_1.full_well_depth == 1e5
        assert ccd_phase_2.well_notch_depth == 0.01
        assert ccd_phase_2.full_well_depth == 2e5
        assert ccd_phase_3.well_notch_depth == 0.01
        assert ccd_phase_3.full_well_depth == 3e5
