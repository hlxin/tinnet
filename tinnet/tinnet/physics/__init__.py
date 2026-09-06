"""TinNet theory ("physics") modules.

Every theory module follows the :class:`PhysicsModel` contract::

    GNN latent outputs + tabulated constants --> physical parameters --> property

Available modules
-----------------
``newns_anderson``   adsorption energy for any registered :class:`AdsorbateSpec`
``cohesion``         cohesive energy (renormalized-atom theory)
``rectangular_band`` d-band filling / center / width of a site
``moments``          second, third, fourth d-band moments from tight binding

Quick start
-----------
>>> from tinnet.tinnet.physics import NewnsAndersonModel, adsorbate_spec
>>> physics = NewnsAndersonModel(adsorbate_spec('OH'))
>>> print(physics.describe())
"""

from .base import PhysicsModel, PHYSICS_REGISTRY, available_physics, get_physics, register_physics
from .adsorbates import (
    ADSORBATE_REGISTRY,
    AdsorbateSpec,
    OrbitalSpec,
    OH_ATOP,
    O_ATOP,
    adsorbate_spec,
    available_adsorbates,
    iter_adsorbates,
    register_adsorbate,
)
from .newns_anderson import NewnsAndersonModel, semi_ellipse_dos, hilbert_transform, hilbert_multiplier
from .cohesion import CohesionModel
from .rectangular_band import RectangularBandModel, hopping_couplings
from .moments import MomentModel
from .shared_adsorption import SharedAdsorptionPhysics

__all__ = [
    "PhysicsModel", "PHYSICS_REGISTRY", "available_physics", "get_physics", "register_physics",
    "ADSORBATE_REGISTRY", "AdsorbateSpec", "OrbitalSpec", "OH_ATOP", "O_ATOP",
    "adsorbate_spec", "available_adsorbates", "iter_adsorbates", "register_adsorbate",
    "NewnsAndersonModel", "semi_ellipse_dos", "hilbert_transform", "hilbert_multiplier",
    "CohesionModel", "RectangularBandModel", "hopping_couplings", "MomentModel",
    "SharedAdsorptionPhysics",
]
