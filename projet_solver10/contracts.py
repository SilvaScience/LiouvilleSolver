"""Public contracts between QuDPy-FDGF and external physical models.

This module defines the accepted operator, state, thermodynamic, collapse,
and sector-model interfaces. These contracts keep physical construction in
the model while giving every backend a stable representation of inputs,
initial conditions, and transition blocks.
"""

from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Protocol, TypeAlias, runtime_checkable

import numpy as np


Sector = Hashable


@runtime_checkable
class OperatorLike(Protocol):
    """Minimal linear-operator interface accepted by backends."""

    shape: tuple
    dtype: Any

    def matvec(self, vector):
        """Vector -> operator action."""

    def rmatvec(self, vector):
        """Vector -> adjoint operator action."""


@dataclass(frozen=True)
class PureState:
    """Pure initial state supplied by an external model."""

    sector: Sector
    vector: Any
    energy: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def normalized_vector(self, *, tolerance=1e-14):
        """State vector -> normalized complex copy."""
        vector = np.asarray(self.vector, dtype=np.complex128).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= tolerance:
            raise ValueError("Initial state vector must have a nonzero norm.")
        return vector / norm


InitialState = PureState


@dataclass(frozen=True)
class DenseDensityBlock:
    """Dense ``rho[ket_sector, bra_sector]`` block of a density matrix."""

    matrix: Any

    def as_matrix(self):
        matrix = np.asarray(self.matrix, dtype=np.complex128)
        if matrix.ndim != 2:
            raise ValueError("A dense density block must be a matrix.")
        return matrix


@dataclass(frozen=True)
class LowRankDensityBlock:
    """Density block represented as ``X S Y†``."""

    X: Any
    Y: Any
    S: Any = None

    def factors(self):
        X = np.asarray(self.X, dtype=np.complex128)
        Y = np.asarray(self.Y, dtype=np.complex128)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be factor matrices.")
        S = (
            np.eye(X.shape[1], Y.shape[1], dtype=np.complex128)
            if self.S is None
            else np.asarray(self.S, dtype=np.complex128)
        )
        if S.shape != (X.shape[1], Y.shape[1]):
            raise ValueError(
                "S must have shape (X_columns, Y_columns)."
            )
        return X, S, Y


DensityBlock: TypeAlias = DenseDensityBlock | LowRankDensityBlock | Any


@dataclass(frozen=True)
class DensityState:
    """Mixed initial state -> sector-based ket/bra blocks.

    Cross-sector blocks may represent bright--dark or momentum coherences.
    """

    blocks: Mapping[tuple[Sector, Sector], DensityBlock]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        blocks = dict(self.blocks)
        if not blocks:
            raise ValueError("DensityState must contain at least one block.")
        normalized = {}
        for key, block in blocks.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError(
                    "Each DensityState key must be (ket_sector, bra_sector)."
                )
            normalized[tuple(key)] = block
        object.__setattr__(self, "blocks", normalized)
        object.__setattr__(self, "metadata", dict(self.metadata))


InitialCondition: TypeAlias = PureState | DensityState


@dataclass(frozen=True)
class ThermodynamicContext:
    """Thermodynamic choices passed to the model without interpretation.
    """

    temperature: float
    ensemble: str = "canonical"
    chemical_potentials: Mapping[str, float] = field(default_factory=dict)
    k_ensemble: str = "independent"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        temperature = float(self.temperature)
        ensemble = str(self.ensemble).lower()
        k_ensemble = str(self.k_ensemble).lower()
        if temperature < 0:
            raise ValueError("Temperature must be non-negative.")
        if ensemble not in {"canonical", "grand_canonical"}:
            raise ValueError(
                "ensemble must be 'canonical' or 'grand_canonical'."
            )
        if k_ensemble not in {"independent", "global"}:
            raise ValueError(
                "k_ensemble must be 'independent' or 'global'."
            )
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "ensemble", ensemble)
        object.__setattr__(self, "k_ensemble", k_ensemble)
        object.__setattr__(
            self,
            "chemical_potentials",
            {
                str(name): float(value)
                for name, value in dict(self.chemical_potentials).items()
            },
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_value(cls, value):
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError(
            "Thermodynamic context must be ThermodynamicContext, mapping, "
            "or None."
        )


@dataclass(frozen=True)
class CollapseChannel:
    """Model-provided ``(L, gamma)`` -> explicit GKSL channel.

    Blocks follow ``{source_sector: {target_sector: operator}}``.
    """

    name: str
    rate: float
    operator_blocks: Mapping[Sector, Mapping[Sector, Any]]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        rate = complex(self.rate)
        if abs(rate.imag) > 1e-14 or rate.real < 0:
            raise ValueError(
                "A Lindblad channel rate must be real and non-negative."
            )
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "rate", float(rate.real))
        object.__setattr__(
            self,
            "operator_blocks",
            {
                source: dict(blocks)
                for source, blocks in dict(self.operator_blocks).items()
            },
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@runtime_checkable
class SectorModel(Protocol):
    """Contract for an external sector-based model.

    Operator blocks may be dense, sparse, or matrix-free.
    """

    def sectors(self):
        """Return sectors in the truncated Hilbert space."""

    def dimension(self, sector: Sector) -> int:
        """Sector -> Hilbert-space dimension."""

    def hamiltonian_blocks(self, source: Sector):
        """Source sector -> ``{target_sector: operator}``."""

    def transition_blocks(
        self,
        operator_name: str,
        direction: str,
        source: Sector,
    ):
        """Transition name and source -> operator blocks."""

    def observable_blocks(self, observable_name: str, source: Sector):
        """Detection name and source -> operator blocks."""

    def initial_condition(self, context=None) -> InitialCondition:
        """Thermodynamic context -> pure or mixed initial state."""

    def equilibrium_state(
        self,
        context: ThermodynamicContext,
    ) -> DensityState:
        """Thermodynamic context -> exact thermal state in the model basis."""

    def collapse_channels(self, context=None):
        """Return explicit Lindblad channels ``(L, gamma)``."""

    def capabilities(self):
        """Return model capabilities."""
