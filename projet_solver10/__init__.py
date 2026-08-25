"""Public interface for the SolverV10 spectroscopy engine.

The package exposes model contracts, propagation protocols, pathway tools,
observable definitions, numerical backends, and result containers. Its main
entry point is ``SpectroscopySolver``, which connects these pieces without
embedding system-specific physics in the solver itself.
"""

from .capabilities import (
    BackendCapabilities,
    Capabilities,
    ModelRequirements,
)
from .contracts import (
    CollapseChannel,
    DenseDensityBlock,
    DensityState,
    InitialCondition,
    InitialState,
    LowRankDensityBlock,
    OperatorLike,
    PureState,
    Sector,
    SectorModel,
    ThermodynamicContext,
)
from .exceptions import (
    CapabilityError,
    ConvergenceError,
    ModelContractError,
    SectorError,
    SolverV10Error,
)
from .pathways import (
    FrequencyPathway,
    Interaction,
    coherence_orders_from_interactions,
    translate_ufss_diagrams,
)
from .protocols import (
    PropagationInterval,
    SpectroscopyProtocol,
    standard_nq_protocol,
)
from .results import PathwayResult, PlotResult, SpectrumResult
from .observables import ObservableSpec, normalize_observables
from .plotting import SpectroscopyPlotter
from .solver import SpectroscopySolver
from .model_adapters import EigenbasisKModel
from .generators import EvolutionGenerator

__all__ = [
    "BackendCapabilities",
    "Capabilities",
    "CapabilityError",
    "CollapseChannel",
    "ConvergenceError",
    "DenseDensityBlock",
    "DensityState",
    "EigenbasisKModel",
    "EvolutionGenerator",
    "FrequencyPathway",
    "InitialCondition",
    "InitialState",
    "Interaction",
    "LowRankDensityBlock",
    "ModelRequirements",
    "ModelContractError",
    "OperatorLike",
    "ObservableSpec",
    "PathwayResult",
    "PlotResult",
    "PropagationInterval",
    "PureState",
    "Sector",
    "SectorError",
    "SectorModel",
    "SolverV10Error",
    "SpectroscopyProtocol",
    "SpectroscopySolver",
    "SpectroscopyPlotter",
    "SpectrumResult",
    "ThermodynamicContext",
    "coherence_orders_from_interactions",
    "standard_nq_protocol",
    "translate_ufss_diagrams",
    "normalize_observables",
]

__version__ = "10.0.0-dev3"
