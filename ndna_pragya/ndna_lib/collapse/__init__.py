# ndna_lib/collapse/__init__.py
from __future__ import annotations

from .config import (
    CollapseConfig,
    BreedingConfig,
    GeometryConfig,
    ProtocolType,
)

from .geometry_runner import (
    run_method5_unified_on_alpaca,
    run_spectral_curvature_on_alpaca,
    save_geometry_metrics,
)

from .protocols import (
    CollapseProtocol,
    CrossBreedingProtocol,
    InbreedingProtocol,
)

__all__ = [
    # config
    "CollapseConfig",
    "BreedingConfig",
    "GeometryConfig",
    "ProtocolType",
    # geometry
    "run_method5_unified_on_alpaca",
    "run_spectral_curvature_on_alpaca",
    "save_geometry_metrics",
    # protocols
    "CollapseProtocol",
    "CrossBreedingProtocol",
    "InbreedingProtocol",
]
