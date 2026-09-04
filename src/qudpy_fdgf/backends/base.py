"""Shared infrastructure for QuDPy-FDGF numerical backends.

This module validates external models, arranges Hilbert-space sectors, and
normalizes operator actions across dense, sparse, and matrix-free inputs. It
also implements common observable and transition handling so concrete
backends can focus on propagation and resolvent algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..capabilities import BackendCapabilities, Capabilities
from ..contracts import (
    CollapseChannel,
    DenseDensityBlock,
    DensityState,
    LowRankDensityBlock,
    PureState,
    ThermodynamicContext,
)
from ..exceptions import ModelContractError, SectorError
from ..generators import EvolutionGenerator
from ..observables import ObservableSpec


def apply_operator(operator, vector):
    """Array, sparse matrix, or operator and vector -> operator action."""
    if hasattr(operator, "matvec"):
        return np.asarray(operator.matvec(vector), dtype=np.complex128)
    return np.asarray(operator @ vector, dtype=np.complex128)


def apply_adjoint_operator(operator, vector):
    """Array, sparse matrix, or operator and vector -> adjoint action."""
    if hasattr(operator, "rmatvec"):
        return np.asarray(operator.rmatvec(vector), dtype=np.complex128)
    if hasattr(operator, "getH"):
        return np.asarray(operator.getH() @ vector, dtype=np.complex128)
    return np.asarray(operator.conj().T @ vector, dtype=np.complex128)


def operator_shape(operator):
    """External operator -> validated shape."""
    shape = getattr(operator, "shape", None)
    if shape is None or len(shape) != 2:
        raise ModelContractError("Every operator must expose shape=(m, n).")
    return tuple(int(item) for item in shape)


@dataclass(frozen=True)
class SectorLayout:
    """Deterministic sector layout within a concatenated vector."""

    sectors: tuple
    dimensions: dict
    slices: dict
    total_dimension: int

    @classmethod
    def from_model(cls, model):
        sectors = tuple(model.sectors())
        if not sectors:
            raise ModelContractError("Model does not provide any sector.")
        if len(set(sectors)) != len(sectors):
            raise ModelContractError("Model sectors must be unique.")

        dimensions = {}
        slices = {}
        offset = 0
        for sector in sectors:
            dimension = int(model.dimension(sector))
            if dimension <= 0:
                raise ModelContractError(
                    f"Invalid dimension {dimension} for sector {sector!r}."
                )
            dimensions[sector] = dimension
            slices[sector] = slice(offset, offset + dimension)
            offset += dimension
        return cls(sectors, dimensions, slices, offset)

    def empty(self):
        return np.zeros(self.total_dimension, dtype=np.complex128)

    def sector_view(self, vector, sector):
        return vector[self.slices[sector]]

    def embed(self, sector, vector):
        if sector not in self.slices:
            raise SectorError(f"Unknown initial sector: {sector!r}")
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        expected = self.dimensions[sector]
        if vector.size != expected:
            raise ModelContractError(
                f"State for sector {sector!r} has size {vector.size}; "
                f"expected {expected}."
            )
        result = self.empty()
        result[self.slices[sector]] = vector
        return result


class BackendBase(ABC):
    """Base class for numerical backends."""

    name = "base"
    capabilities = Capabilities()
    backend_capabilities = BackendCapabilities(
        state_kinds=("pure",),
        supported_pairs=(("unitary", "time"),),
        exactness="exact",
        matrix_free=True,
    )

    def __init__(self, *, eta=0.05, **options):
        eta = float(eta)
        if eta < 0:
            raise ValueError("eta must be non-negative.")
        self.eta = eta
        self.options = dict(options)
        self.model = None
        self.model_capabilities = Capabilities()
        self.layout = None
        self.initial_state = None
        self.initial_condition = None
        self._initial_vector = None
        self._initial_density_matrix = None
        self._initial_is_pure = False
        self._hamiltonian_blocks = {}
        self._collapse_channels = ()
        self._collapse_operators = ()
        self._collapse_channel_map = {}
        self._transition_blocks = {}
        self._observable_blocks = {}
        self._hamiltonian = None
        self.generator = None

    def build(self, model, context=None):
        """Model and context -> initialized sector operators and state."""
        required = (
            "sectors",
            "dimension",
            "hamiltonian_blocks",
            "transition_blocks",
            "observable_blocks",
        )
        missing = [
            name for name in required if not callable(getattr(model, name, None))
        ]
        if missing:
            raise ModelContractError(
                "Model violates the contract; missing methods: "
                + ", ".join(missing)
            )

        self.model = model
        self.context = ThermodynamicContext.from_value(context)
        capabilities = getattr(model, "capabilities", None)
        capabilities = capabilities() if callable(capabilities) else capabilities
        self.model_capabilities = Capabilities.from_value(capabilities)
        self.layout = SectorLayout.from_model(model)
        self._cache_and_validate_hamiltonian()
        self._hamiltonian = LinearOperator(
            shape=(
                self.layout.total_dimension,
                self.layout.total_dimension,
            ),
            dtype=np.complex128,
            matvec=self._hamiltonian_matvec,
            rmatvec=self._hamiltonian_rmatvec,
        )
        self._cache_and_validate_collapse_channels()
        self.generator = EvolutionGenerator(
            self._hamiltonian,
            self._collapse_operators,
        )
        self._load_initial_condition()
        return self

    def _call_with_optional_context(self, method):
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            parameters = None
        if parameters is not None and not parameters:
            return method()
        return method(self.context)

    def _resolve_initial_condition(self):
        if self.context is not None:
            equilibrium = getattr(self.model, "equilibrium_state", None)
            if callable(equilibrium):
                return equilibrium(self.context)

        initial_condition = getattr(self.model, "initial_condition", None)
        if callable(initial_condition):
            return self._call_with_optional_context(initial_condition)

        initial_state = getattr(self.model, "initial_state", None)
        if callable(initial_state):
            return initial_state()
        raise ModelContractError(
            "Model must expose initial_condition(context), "
            "equilibrium_state(context), or initial_state()."
        )

    def _density_block_matrix(self, block):
        if isinstance(block, DenseDensityBlock):
            return block.as_matrix()
        if isinstance(block, LowRankDensityBlock):
            X, S, Y = block.factors()
            return X @ S @ Y.conj().T
        if hasattr(block, "toarray"):
            return np.asarray(block.toarray(), dtype=np.complex128)
        matrix = np.asarray(block, dtype=np.complex128)
        if matrix.ndim != 2:
            raise ModelContractError(
                "Every DensityState block must be a matrix."
            )
        return matrix

    def _assemble_density_state(self, state):
        dimension = self.layout.total_dimension
        density = np.zeros((dimension, dimension), dtype=np.complex128)
        for (ket_sector, bra_sector), block in state.blocks.items():
            if ket_sector not in self.layout.dimensions:
                raise SectorError(
                    f"Unknown initial ket sector: {ket_sector!r}"
                )
            if bra_sector not in self.layout.dimensions:
                raise SectorError(
                    f"Unknown initial bra sector: {bra_sector!r}"
                )
            matrix = self._density_block_matrix(block)
            expected = (
                self.layout.dimensions[ket_sector],
                self.layout.dimensions[bra_sector],
            )
            if matrix.shape != expected:
                raise ModelContractError(
                    f"Density block {(ket_sector, bra_sector)!r} has shape "
                    f"{matrix.shape}; expected {expected}."
                )
            density[
                self.layout.slices[ket_sector],
                self.layout.slices[bra_sector],
            ] = matrix
        return density

    def _validate_physical_density(self, density):
        tolerance = float(
            self.options.get("density_matrix_tolerance", 1e-10)
        )
        if not np.allclose(
            density, density.conj().T, atol=tolerance, rtol=0.0
        ):
            raise ModelContractError(
                "Initial density matrix is not Hermitian."
            )
        trace = np.trace(density)
        if abs(trace.imag) > tolerance or not np.isclose(
            trace.real, 1.0, atol=tolerance, rtol=0.0
        ):
            raise ModelContractError(
                "Initial density matrix must have unit trace; "
                f"got {trace}."
            )
        eigenvalues = np.linalg.eigvalsh(
            0.5 * (density + density.conj().T)
        )
        if float(np.min(eigenvalues)) < -tolerance:
            raise ModelContractError(
                "Initial density matrix is not positive "
                f"(smallest eigenvalue={np.min(eigenvalues):.3e})."
            )

    def _load_initial_condition(self):
        condition = self._resolve_initial_condition()
        if not isinstance(condition, (PureState, DensityState)):
            if isinstance(condition, tuple) and len(condition) in {2, 3}:
                condition = PureState(*condition)
            else:
                raise ModelContractError(
                    "Initial condition must be PureState, DensityState, "
                    "or (sector, vector[, energy])."
                )

        self.initial_condition = condition
        if isinstance(condition, PureState):
            vector = condition.normalized_vector()
            self.initial_state = PureState(
                condition.sector,
                vector,
                condition.energy,
                dict(condition.metadata),
            )
            self._initial_vector = self.layout.embed(
                condition.sector, vector
            )
            self._initial_density_matrix = np.outer(
                self._initial_vector, self._initial_vector.conj()
            )
            self._initial_is_pure = True
            return

        density = self._assemble_density_state(condition)
        self._validate_physical_density(density)
        self.initial_state = condition
        self._initial_vector = None
        self._initial_density_matrix = density
        self._initial_is_pure = False

    def _normalize_blocks(self, blocks, *, source, family):
        if blocks is None:
            return {}
        try:
            blocks = dict(blocks)
        except (TypeError, ValueError) as exc:
            raise ModelContractError(
                f"{family}({source!r}) must return a mapping."
            ) from exc
        normalized = {}
        source_dimension = self.layout.dimensions[source]
        for target, operator in blocks.items():
            if target not in self.layout.dimensions:
                raise SectorError(
                    f"{family} connects {source!r} to unknown sector {target!r}."
                )
            expected = (
                self.layout.dimensions[target],
                source_dimension,
            )
            actual = operator_shape(operator)
            if actual != expected:
                raise ModelContractError(
                    f"{family} block {source!r}->{target!r} has shape {actual}; "
                    f"expected shape {expected}."
                )
            normalized[target] = operator
        return normalized

    def _cache_and_validate_hamiltonian(self):
        for source in self.layout.sectors:
            blocks = self.model.hamiltonian_blocks(source)
            self._hamiltonian_blocks[source] = self._normalize_blocks(
                blocks,
                source=source,
                family="hamiltonian_blocks",
            )

    def _hamiltonian_matvec(self, vector):
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        if vector.size != self.layout.total_dimension:
            raise ValueError("Size is incompatible with the Hilbert space.")
        result = self.layout.empty()
        for source, blocks in self._hamiltonian_blocks.items():
            source_vector = self.layout.sector_view(vector, source)
            for target, operator in blocks.items():
                result[self.layout.slices[target]] += apply_operator(
                    operator, source_vector
                )
        return result

    def _hamiltonian_rmatvec(self, vector):
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        if vector.size != self.layout.total_dimension:
            raise ValueError("Size is incompatible with the Hilbert space.")
        result = self.layout.empty()
        for source, blocks in self._hamiltonian_blocks.items():
            for target, operator in blocks.items():
                target_vector = self.layout.sector_view(vector, target)
                result[self.layout.slices[source]] += apply_adjoint_operator(
                    operator, target_vector
                )
        return result

    def _block_operator_matvec(self, blocks_by_source, vector):
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        result = self.layout.empty()
        for source, blocks in blocks_by_source.items():
            source_vector = self.layout.sector_view(vector, source)
            for target, operator in blocks.items():
                result[self.layout.slices[target]] += apply_operator(
                    operator, source_vector
                )
        return result

    def _block_operator_rmatvec(self, blocks_by_source, vector):
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        result = self.layout.empty()
        for source, blocks in blocks_by_source.items():
            for target, operator in blocks.items():
                target_vector = self.layout.sector_view(vector, target)
                result[self.layout.slices[source]] += apply_adjoint_operator(
                    operator, target_vector
                )
        return result

    def _cache_and_validate_collapse_channels(self):
        method = getattr(self.model, "collapse_channels", None)
        raw_channels = (
            self._call_with_optional_context(method)
            if callable(method)
            else ()
        )
        channels = []
        operators = []
        for raw_channel in raw_channels or ():
            channel = (
                raw_channel
                if isinstance(raw_channel, CollapseChannel)
                else CollapseChannel(**raw_channel)
                if isinstance(raw_channel, dict)
                else None
            )
            if channel is None:
                raise ModelContractError(
                    "collapse_channels() must return CollapseChannel objects "
                    "or compatible mappings."
                )
            normalized = {}
            for source, blocks in channel.operator_blocks.items():
                if source not in self.layout.dimensions:
                    raise SectorError(
                        f"Channel {channel.name!r}: unknown source sector "
                        f"{source!r}."
                    )
                normalized[source] = self._normalize_blocks(
                    blocks,
                    source=source,
                    family=f"collapse_channel[{channel.name}]",
                )
            dimension = self.layout.total_dimension
            operator = LinearOperator(
                (dimension, dimension),
                dtype=np.complex128,
                matvec=lambda value, blocks=normalized: (
                    self._block_operator_matvec(blocks, value)
                ),
                rmatvec=lambda value, blocks=normalized: (
                    self._block_operator_rmatvec(blocks, value)
                ),
            )
            channels.append(channel)
            operators.append((channel, operator))
        self._collapse_channels = tuple(channels)
        self._collapse_operators = tuple(operators)
        self._collapse_channel_map = {
            channel.name: (channel, operator)
            for channel, operator in self._collapse_operators
        }

    def jump_channel_names(self):
        """Return GKSL channel names available for jump counting."""
        return tuple(channel.name for channel in self._collapse_channels)

    def apply_jump_effect(self, vector, channel_name):
        """Channel and Hilbert vector -> ``gamma L^dagger L`` action."""
        try:
            channel, operator = self._collapse_channel_map[str(channel_name)]
        except KeyError as exc:
            available = ", ".join(self.jump_channel_names()) or "none"
            raise KeyError(
                f"Unknown GKSL channel {channel_name!r}; available: {available}."
            ) from exc
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        return channel.rate * apply_adjoint_operator(
            operator,
            apply_operator(operator, vector),
        )

    def apply_detection_operator(self, vector, observable):
        """State and observable -> final contraction or jump rate."""
        spec = (
            observable
            if isinstance(observable, ObservableSpec)
            else ObservableSpec(name=str(observable))
        )
        if spec.kind == "operator":
            return self.apply_observable(vector, spec.operator)
        if spec.kind == "jump_rate":
            return self.apply_jump_effect(vector, spec.channel)
        raise ValueError(
            "An integrated_jump observable requires time-window evaluation, "
            "not a final contraction."
        )

    def _get_transition_blocks(self, operator_name, direction, source):
        direction = str(direction).lower()
        if direction not in {"plus", "minus"}:
            raise ValueError("direction must be 'plus' or 'minus'.")
        key = (str(operator_name), direction, source)
        if key not in self._transition_blocks:
            blocks = self.model.transition_blocks(*key)
            self._transition_blocks[key] = self._normalize_blocks(
                blocks,
                source=source,
                family=f"transition_blocks[{operator_name},{direction}]",
            )
        return self._transition_blocks[key]

    def apply_transition(
        self,
        vector,
        *,
        operator_name,
        direction,
        source_sector=None,
        target_sector=None,
    ):
        """Sector vector and transition -> concatenated target vector."""
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        if vector.size != self.layout.total_dimension:
            raise ValueError("Size is incompatible with the Hilbert space.")
        result = self.layout.empty()
        matched_block = False
        sources = (
            (source_sector,)
            if source_sector is not None
            else self.layout.sectors
        )
        for source in sources:
            if source not in self.layout.dimensions:
                raise SectorError(f"Unknown source sector: {source!r}")
            source_vector = self.layout.sector_view(vector, source)
            blocks = self._get_transition_blocks(
                operator_name, direction, source
            )
            for target, operator in blocks.items():
                if target_sector is not None and target != target_sector:
                    continue
                matched_block = True
                result[self.layout.slices[target]] += apply_operator(
                    operator, source_vector
                )
        if (
            target_sector is not None
            and target_sector not in self.layout.dimensions
        ):
            raise SectorError(f"Unknown target sector: {target_sector!r}")
        if (
            source_sector is not None or target_sector is not None
        ) and not matched_block:
            raise SectorError(
                "Constrained transition matches no model block: "
                f"operator={operator_name!r}, direction={direction!r}, "
                f"source={source_sector!r}, target={target_sector!r}."
            )
        return result

    def _get_observable_blocks(self, observable_name, source):
        key = (str(observable_name), source)
        if key not in self._observable_blocks:
            blocks = self.model.observable_blocks(*key)
            self._observable_blocks[key] = self._normalize_blocks(
                blocks,
                source=source,
                family=f"observable_blocks[{observable_name}]",
            )
        return self._observable_blocks[key]

    def apply_observable(self, vector, observable_name):
        """Concatenated vector and observable -> observable action."""
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        if vector.size != self.layout.total_dimension:
            raise ValueError("Size is incompatible with the Hilbert space.")
        result = self.layout.empty()
        for source in self.layout.sectors:
            source_vector = self.layout.sector_view(vector, source)
            for target, operator in self._get_observable_blocks(
                observable_name, source
            ).items():
                result[self.layout.slices[target]] += apply_operator(
                    operator, source_vector
                )
        return result

    def materialize(self, operator):
        """Operator action -> dense matrix for small spaces only."""
        rows, columns = operator_shape(operator)
        matrix = np.empty((rows, columns), dtype=np.complex128)
        basis_vector = np.zeros(columns, dtype=np.complex128)
        for index in range(columns):
            basis_vector[index] = 1.0
            matrix[:, index] = apply_operator(operator, basis_vector)
            basis_vector[index] = 0.0
        return matrix

    def clear_caches(self):
        """Clear coordinate-dependent caches."""

    @abstractmethod
    def calc_pathway(self, pathway, protocol, coordinates):
        """Pathway, protocol, and coordinates -> pathway response."""
