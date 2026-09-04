"""Static many-body services kept separate from spectroscopy propagation.

The existing QuDPy-FDGF backends operate on explicit Hilbert/Liouville states.
Tensor-network ground-state engines instead return compressed-state
observables through this independent contract.  A future dynamical MPS
backend can build on the same registry without pretending that an MPS is a
dense ``PureState``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SpinChainGroundStateProblem:
    """Open dimerized spin-half Heisenberg-chain request."""

    number_of_sites: int
    J1_eV: float
    delta: float = 0.0
    J2_eV: float = 0.0
    boundary: str = "open"

    def __post_init__(self):
        if self.number_of_sites < 2 or self.number_of_sites % 2:
            raise ValueError("number_of_sites must be an even integer >= 2.")
        if self.J1_eV <= 0.0:
            raise ValueError("J1_eV must be positive.")
        if abs(self.delta) > 1.0:
            raise ValueError("|delta| must not exceed 1.")
        if self.J2_eV != 0.0:
            raise NotImplementedError(
                "The first static DMRG contract freezes J2_eV=0."
            )
        if self.boundary != "open":
            raise NotImplementedError(
                "The first static DMRG contract supports open chains only."
            )


@dataclass(frozen=True)
class DMRGSettings:
    """Numerical settings whose convergence must be reported."""

    chi_max: int = 256
    svd_min: float = 1.0e-12
    max_sweeps: int = 30
    max_energy_error_eV: float = 1.0e-12
    mixer: bool = True

    def __post_init__(self):
        if self.chi_max < 2:
            raise ValueError("chi_max must be at least 2.")
        if self.svd_min < 0.0:
            raise ValueError("svd_min must be nonnegative.")
        if self.max_sweeps < 1:
            raise ValueError("max_sweeps must be positive.")


@dataclass
class ManyBodyGroundStateResult:
    """Serializable static output shared by exact and MPS engines."""

    engine: str
    problem: SpinChainGroundStateProblem
    settings: DMRGSettings
    energy_sz0_eV: float
    energy_sz1_eV: float
    gap_eV: float
    bond_correlations: list[float]
    C1_all: float
    D_all: float
    C1_bulk: float
    D_bulk: float
    correlation_length_proxy_sites: float | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "problem": asdict(self.problem),
            "settings": asdict(self.settings),
            "energy_sz0_eV": float(self.energy_sz0_eV),
            "energy_sz1_eV": float(self.energy_sz1_eV),
            "gap_eV": float(self.gap_eV),
            "bond_correlations": [float(value) for value in self.bond_correlations],
            "C1_all": float(self.C1_all),
            "D_all": float(self.D_all),
            "C1_bulk": float(self.C1_bulk),
            "D_bulk": float(self.D_bulk),
            "correlation_length_proxy_sites": (
                None
                if self.correlation_length_proxy_sites is None
                else float(self.correlation_length_proxy_sites)
            ),
            "diagnostics": dict(self.diagnostics),
        }


class ManyBodyGroundStateEngine(Protocol):
    """Minimal interface implemented by static tensor-network engines."""

    name: str

    def solve(
        self,
        problem: SpinChainGroundStateProblem,
        settings: DMRGSettings,
    ) -> ManyBodyGroundStateResult: ...


class ManyBodySolver:
    """Small registry/orchestrator independent of ``SpectroscopySolver``."""

    _ENGINE_CLASSES: dict[str, type] = {}

    def __init__(self, engine: str = "tenpy_dmrg"):
        self.engine_name = str(engine).lower()

    @classmethod
    def register_engine(cls, name: str, engine_class: type) -> None:
        normalized = str(name).lower()
        if not normalized:
            raise ValueError("The many-body engine name cannot be empty.")
        if not callable(engine_class):
            raise TypeError("engine_class must be constructible.")
        cls._ENGINE_CLASSES[normalized] = engine_class

    @classmethod
    def available_engines(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._ENGINE_CLASSES))

    def solve(
        self,
        problem: SpinChainGroundStateProblem,
        settings: DMRGSettings | None = None,
    ) -> ManyBodyGroundStateResult:
        if not isinstance(problem, SpinChainGroundStateProblem):
            raise TypeError("problem must be a SpinChainGroundStateProblem.")
        settings = DMRGSettings() if settings is None else settings
        if not isinstance(settings, DMRGSettings):
            raise TypeError("settings must be DMRGSettings.")
        try:
            engine_class = self._ENGINE_CLASSES[self.engine_name]
        except KeyError as exc:
            supported = ", ".join(self.available_engines()) or "none"
            raise ValueError(
                f"Unknown many-body engine {self.engine_name!r}; "
                f"available engines: {supported}."
            ) from exc
        return engine_class().solve(problem, settings)
