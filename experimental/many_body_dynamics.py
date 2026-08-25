"""Contracts for dynamical many-body calculations.

This module defines spin-correlation requests, real-time MPS settings, result
containers, and engine protocols independently of dense spectroscopy
backends. The separation lets tensor-network solvers expose compressed
dynamics without pretending to return explicit Hilbert-space states.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .many_body import DMRGSettings, SpinChainGroundStateProblem


@dataclass(frozen=True)
class SpinCorrelationProblem:
    """Connected zero-temperature spin correlation requested from an MPS."""

    ground_state: SpinChainGroundStateProblem
    operator_geometry: str
    spin_component: str = "Sz"

    def __post_init__(self):
        if self.operator_geometry not in {"local", "q0", "qpi"}:
            raise ValueError("operator_geometry must be local, q0, or qpi.")
        if self.spin_component != "Sz":
            raise NotImplementedError(
                "Test 22d first validates the charge-conserving Sz channel."
            )


@dataclass(frozen=True)
class RealTimeMPSSettings:
    """Numerical controls for a finite real-time MPS trajectory."""

    dmrg: DMRGSettings = field(default_factory=DMRGSettings)
    dt_eV_inverse: float = 2.0
    number_of_steps: int = 64
    chi_max: int = 256
    svd_min: float = 1.0e-12
    tdvp_sites: int = 2

    def __post_init__(self):
        if self.dt_eV_inverse <= 0.0:
            raise ValueError("dt_eV_inverse must be positive.")
        if self.number_of_steps < 1:
            raise ValueError("number_of_steps must be positive.")
        if self.chi_max < 2:
            raise ValueError("chi_max must be at least 2.")
        if self.svd_min < 0.0:
            raise ValueError("svd_min must be nonnegative.")
        if self.tdvp_sites not in (1, 2):
            raise ValueError("tdvp_sites must be 1 or 2.")


@dataclass
class SpinCorrelationResult:
    """Serializable connected correlation and evolution diagnostics."""

    engine: str
    problem: SpinCorrelationProblem
    settings: RealTimeMPSSettings
    times_eV_inverse: list[float]
    correlation_real: list[float]
    correlation_imag: list[float]
    prepared_norm: float
    ground_energy_eV: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def correlation(self):
        import numpy as np

        return np.asarray(self.correlation_real) + 1.0j * np.asarray(
            self.correlation_imag
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "problem": {
                "ground_state": asdict(self.problem.ground_state),
                "operator_geometry": self.problem.operator_geometry,
                "spin_component": self.problem.spin_component,
            },
            "settings": asdict(self.settings),
            "times_eV_inverse": [float(value) for value in self.times_eV_inverse],
            "correlation_real": [float(value) for value in self.correlation_real],
            "correlation_imag": [float(value) for value in self.correlation_imag],
            "prepared_norm": float(self.prepared_norm),
            "ground_energy_eV": float(self.ground_energy_eV),
            "diagnostics": dict(self.diagnostics),
        }


class ManyBodyDynamicsEngine(Protocol):
    name: str

    def correlate(
        self,
        problem: SpinCorrelationProblem,
        settings: RealTimeMPSSettings,
    ) -> SpinCorrelationResult: ...


class ManyBodyDynamicsSolver:
    """Registry for tensor-network dynamics, separate from Liouville solvers."""

    _ENGINE_CLASSES: dict[str, type] = {}

    def __init__(self, engine: str = "tenpy_tdvp"):
        self.engine_name = str(engine).lower()

    @classmethod
    def register_engine(cls, name: str, engine_class: type) -> None:
        normalized = str(name).lower()
        if not normalized:
            raise ValueError("The dynamics engine name cannot be empty.")
        cls._ENGINE_CLASSES[normalized] = engine_class

    @classmethod
    def available_engines(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._ENGINE_CLASSES))

    def correlate(
        self,
        problem: SpinCorrelationProblem,
        settings: RealTimeMPSSettings | None = None,
    ) -> SpinCorrelationResult:
        if not isinstance(problem, SpinCorrelationProblem):
            raise TypeError("problem must be a SpinCorrelationProblem.")
        settings = RealTimeMPSSettings() if settings is None else settings
        if not isinstance(settings, RealTimeMPSSettings):
            raise TypeError("settings must be RealTimeMPSSettings.")
        try:
            engine_class = self._ENGINE_CLASSES[self.engine_name]
        except KeyError as exc:
            supported = ", ".join(self.available_engines()) or "none"
            raise ValueError(
                f"Unknown dynamics engine {self.engine_name!r}; "
                f"available engines: {supported}."
            ) from exc
        return engine_class().correlate(problem, settings)
