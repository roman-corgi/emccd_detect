import numpy as np
from copy import deepcopy

from arcticpy import util
import arcticpy as ac


class CCD(object):
    def __init__(
        self,
        fraction_of_traps_per_phase=[1],
        full_well_depth=1e4,
        well_notch_depth=0.0,
        well_fill_power=0.58,
        well_bloom_level=None,
    ):
        """
        A model describing how electrons fill the volume inside each phase of
        a pixel in a CCD detector.

        By default, each pixel is assumed to have only a single phase. To
        specify a multi-phase device (which will need a corresponding clocking
        sequence to be defined in readout electronics) specify the fraction of
        traps as a list, and optionally different values for the other
        parameters as lists too. If the trap density is uniform, then the ratio
        of full_well_depth to fraction_of_traps_per_phase will be the same in
        all phases.

        All the following are equivalent:
            ccd.cloud_fractional_volumes_from_n_electrons_and_phase(
                n_electrons, phase
            )

            f = ccd.well_filling_function(
                phase
            )
            f(n_electrons)

            p = ac.CCDPhase(ccd, phase)
            p.cloud_fractional_volumes_from_n_electrons(n_electrons)


        Parameters
        ----------
        fraction_of_traps_per_phase : float or [float]
            Assuming that traps have uniform density throughout the CCD, for
            multi-phase clocking, this specifies the physical width of each
            phase. This is used only to distribute the traps between phases.
            The units do not matter and can be anything from microns to light
            years, or fractions of a pixel. Only the fractional widths are ever
            returned. If this is an array then you can optionally also enter
            lists of any/all of the other parameters for each phase.

        full_well_depth : float or [float]
            The maximum number of electrons that can be contained within a
            pixel/phase. For multiphase clocking, if only one value is supplied,
            that is (by default) replicated to all phases. However, different
            physical widths of phases can be set by specifying the full well
            depth as a list containing different values. If the potential in
            more than one phase is held high during any stage in the clocking
            cycle, their full well depths are added together. This value is
            indpependent of the fraction of traps allocated to each phase.

        well_notch_depth : float or [float]
            The number of electrons that fit inside a 'notch' at the bottom of a
            potential well, occupying negligible volume and therefore being
            immune to trapping. These electrons still count towards the full
            well depth. The notch depth can, in  principle, vary between phases.

        well_fill_power : float or [float]
            The exponent in a power-law model of the volume occupied by a cloud
            of electrons. This can, in principle, vary between phases.

        well_bloom_level : float or [float]
            Acts similarly to a notch, but for surface traps.
            Default value is full_well_depth - i.e. no blooming is possible.
        """

        # All parameters are returned as a list of length n_phases
        self.fraction_of_traps_per_phase = fraction_of_traps_per_phase
        self.full_well_depth = full_well_depth
        self.well_fill_power = well_fill_power
        self.well_notch_depth = well_notch_depth
        self.well_bloom_level = well_bloom_level

    @property
    def fraction_of_traps_per_phase(self):
        return self._fraction_of_traps_per_phase

    @fraction_of_traps_per_phase.setter
    def fraction_of_traps_per_phase(self, value):
        if isinstance(value, list):
            self._fraction_of_traps_per_phase = value
        else:
            self._fraction_of_traps_per_phase = [
                value
            ]  # Make sure the arrays are arrays
        self._n_phases = len(self._fraction_of_traps_per_phase)

    @property
    def n_phases(self):
        return self._n_phases

    @property
    def full_well_depth(self):
        return self._full_well_depth

    @full_well_depth.setter
    def full_well_depth(self, value):
        if isinstance(value, list):
            if len(value) != self.n_phases:
                raise ValueError(
                    f"Incorrect number of phases ({len(value)}) in full_well_depth"
                )
            self._full_well_depth = value
        else:
            self._full_well_depth = [value] * self.n_phases

    @property
    def well_fill_power(self):
        return self._well_fill_power

    @well_fill_power.setter
    def well_fill_power(self, value):
        if isinstance(value, list):
            if len(value) != self.n_phases:
                raise ValueError(
                    f"Incorrect number of phases ({len(value)}) in well_fill_power"
                )
            self._well_fill_power = value
        else:
            self._well_fill_power = [value] * self.n_phases

    @property
    def well_notch_depth(self):
        return self._well_notch_depth

    @well_notch_depth.setter
    def well_notch_depth(self, value):
        if isinstance(value, list):
            if len(value) != self.n_phases:
                raise ValueError(
                    f"Incorrect number of phases ({len(value)}) in well_notch_depth"
                )
            self._well_notch_depth = value
        else:
            self._well_notch_depth = [value] * self.n_phases

    @property
    def well_bloom_level(self):
        return self._well_bloom_level

    @well_bloom_level.setter
    def well_bloom_level(self, value):
        if isinstance(value, list):
            if len(value) != self.n_phases:
                raise ValueError(
                    f"Incorrect number of phases ({len(value)}) in well_bloom_level"
                )
            self._well_bloom_level = value
        else:
            if value is None:
                value = self.full_well_depth
            self._well_bloom_level = [value] * self.n_phases

    def cloud_fractional_volumes_from_n_electrons_and_phase(
        self, n_electrons, phase=0, surface=False
    ):
        """Calculate the fractional volume of a charge cloud.

        Parameters
        ----------
        n_electrons : [float]
            The number of electrons in a charge cloud in each column.

        phase : int
            The phase of the pixel.

        Returns
        -------
        cloud_fractional_volume : [float]
            The volume of the charge cloud, as a fraction of the total volume
            of the pixel (or phase), in each column.

        surface : bool
            #
            # RJM: RESERVED FOR SURFACE TRAPS
            #
        """
        return self.well_filling_function(phase)(n_electrons, surface)

    def well_filling_function(self, phase=0):
        """Return a self-contained function describing the well-filling model.

        The returned function calculates the fractional volumes of charge clouds
        in a pixel (and phase) from the number of electrons in the cloud in each
        column, which are used to calculate the proportion of traps that are
        reached by the cloud and can capture charge from it.

        By default, it is assumed that the traps are uniformly distributed
        throughout the volume, but that assumption can be relaxed by adjusting
        this function to reflect the net number of traps seen as a function of
        charge cloud size. An example would be for surface traps, which are
        responsible for blooming (which is asymmetric and happens during
        readout, unlike bleeding) and are preset only at the top of a pixel.

        This function embodies the core assumption of a volume-driven CTI
        model like ArCTIC: that traps are either exposed (and have a
        constant capture timescale, which may be zero for instant capture),
        or unexposed and therefore unavailable. This behaviour differs from
        a density-driven CTI model, in which traps may capture an electron
        anywhere in a pixel, but at varying capture probability. There is
        considerable evidence that CCDs in the Hubble Space Telescope are
        primarily volume-driven; a software algorithm to mimic such behaviour
        also runs much faster.

        Parameters
        ----------
        phase : int
            The phase of the pixel. Multiple phases may optionally have
            different CCD parameters.

        Returns
        -------
        well_filling_function : func
            A self-contained function describing the well-filling model for this
            phase.
        """

        def cloud_fractional_volumes_from_n_electrons(n_electrons, surface=False):
            """Calculate the fractional volume of charge cloud.

            Parameters
            ----------
            n_electrons : [float]
                The number of electrons in a charge cloud in each column.

            surface : bool
                #
                # RJM: RESERVED FOR SURFACE TRAPS
                #

            Returns
            -------
            volumes : [float]
                The fraction of traps of this species exposed in each column.
            """
            if type(n_electrons) is float:
                n_electrons = np.array([n_electrons])
            elif type(n_electrons) is list:
                n_electrons = np.array(n_electrons)

            # Set well parameters for surface or standard traps
            if surface:
                empty = self.blooming_level[phase]
                beta = 1
            else:
                empty = self.well_notch_depth[phase]
                beta = self.well_fill_power[phase]
            well_range = self.full_well_depth[phase] - empty

            volumes = (
                util.set_array_min_max((n_electrons - empty) / well_range, 0, 1)
            ) ** beta

            return volumes

        return cloud_fractional_volumes_from_n_electrons


class CCDPhase(object):
    def __init__(self, ccd=CCD(), phase=0):
        """Extract just one phase from the CCD"""

        self.fraction_of_traps_per_phase = ccd.fraction_of_traps_per_phase[phase]
        self.full_well_depth = ccd.full_well_depth[phase]
        self.well_fill_power = ccd.well_fill_power[phase]
        self.well_notch_depth = ccd.well_notch_depth[phase]
        self.well_bloom_level = ccd.well_bloom_level[phase]
        self.cloud_fractional_volumes_from_n_electrons = ccd.well_filling_function(
            phase
        )
