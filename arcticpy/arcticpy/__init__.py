from autoarray.instruments import acs

from arcticpy.main import add_cti, remove_cti, model_for_HST_ACS
from arcticpy.roe import (
    ROE,
    ROEChargeInjection,
    ROETrapPumping,
)
from arcticpy.ccd import CCD, CCDPhase
from arcticpy.traps import (
    Trap,
    TrapInstantCapture,
    TrapLifetimeContinuumAbstract,
    TrapLogNormalLifetimeContinuum,
)
from arcticpy.trap_managers import (
    AllTrapManager,
    TrapManager,
    TrapManagerTrackTime,
    TrapManagerInstantCapture,
)
