"""Optional model adapters kept outside the solver core.

This module turns site-basis Hamiltonians and interaction operators into
objects that satisfy the ``SectorModel`` contract. It handles optional
momentum slices, diagonalization, transition decomposition, and weighted
initial states. These physical choices stay in the adapter rather than the
solver or numerical backends.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .contracts import (
    CollapseChannel,
    DensityState,
    DenseDensityBlock,
    PureState,
)
from .capabilities import ModelRequirements


_BOLTZMANN_EV_PER_KELVIN = 8.617333262e-5


def _validated_boltzmann_constant(value):
    """Return a positive finite Boltzmann constant in model energy units/K."""
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("boltzmann_constant must be positive and finite.")
    return value


def _as_k_stack(array, name, expected_d=None):
    """2D or 3D operator array -> normalized ``(N_k, d, d)`` stack."""
    array = np.asarray(array, dtype=np.complex128)
    if array.ndim == 2:
        if array.shape[0] != array.shape[1]:
            raise ValueError(f"{name} must be square; got shape {array.shape}.")
        array = array[np.newaxis, :, :]
    elif array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError(
            f"{name} must have shape (d, d) or (N_k, d, d); "
            f"got {array.shape}."
        )
    if expected_d is not None and array.shape[1] != expected_d:
        raise ValueError(
            f"{name} has dimension {array.shape[1]}; expected {expected_d}."
        )
    return array


def _as_sector_hamiltonians(blocks):
    """Sector -> validated dense Hermitian Hamiltonian blocks."""
    if not isinstance(blocks, Mapping) or not blocks:
        raise TypeError("hamiltonian_blocks must be a non-empty mapping.")
    normalized = {}
    for sector, block in blocks.items():
        matrix = np.asarray(block, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Hamiltonian block for sector {sector!r} must be square; "
                f"got shape {matrix.shape}."
            )
        if not np.allclose(matrix, matrix.conj().T, atol=1e-12, rtol=0.0):
            raise ValueError(
                f"Hamiltonian block for sector {sector!r} must be Hermitian."
            )
        normalized[sector] = matrix
    return normalized


def _as_sector_operator_blocks(blocks, name, dimensions):
    """Flat ``(target, source)`` mapping -> validated dense blocks."""
    if not isinstance(blocks, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    normalized = {}
    for key, block in blocks.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(
                f"Every {name} key must be (target_sector, source_sector)."
            )
        target, source = key
        if target not in dimensions:
            raise KeyError(f"Unknown target sector {target!r} in {name}.")
        if source not in dimensions:
            raise KeyError(f"Unknown source sector {source!r} in {name}.")
        matrix = np.asarray(block, dtype=np.complex128)
        expected = (dimensions[target], dimensions[source])
        if matrix.shape != expected:
            raise ValueError(
                f"{name}[{key!r}] has shape {matrix.shape}; expected "
                f"{expected}."
            )
        normalized[(target, source)] = matrix
    return normalized


def _group_sector_operator_blocks(blocks, sectors):
    """Flat operator blocks -> ``source -> {target: block}`` mapping."""
    grouped = {sector: {} for sector in sectors}
    for (target, source), block in blocks.items():
        grouped[source][target] = block
    return grouped


class ExcitationSectorModel:
    """Excitation-manifold blocks -> diagonalized ``SectorModel`` adapter.

    Hamiltonians are supplied as ``{sector: H_sector}``. Raising, lowering,
    detection, observable, and collapse operators use flat mappings with keys
    ``(target_sector, source_sector)``. Sector dimensions may differ.

    When ``lowering_blocks`` is omitted, it is constructed as the blockwise
    adjoint of ``raising_blocks``. The default detection operator is their
    sum. The initial condition is one eigenstate in one sector; if no sector
    is selected explicitly, the adapter chooses the global lowest-energy
    eigenstate. ``boltzmann_constant`` must use the same energy unit as the
    Hamiltonian blocks; its default value is in eV/K.
    """

    def __init__(
        self,
        hamiltonian_blocks,
        raising_blocks,
        *,
        lowering_blocks=None,
        detection_blocks=None,
        observable_op_blocks=None,
        c_ops_raw=(),
        initial_sector=None,
        initial_state_index=0,
        boltzmann_constant=_BOLTZMANN_EV_PER_KELVIN,
    ):
        self._boltzmann_constant = _validated_boltzmann_constant(
            boltzmann_constant
        )
        hamiltonians = _as_sector_hamiltonians(hamiltonian_blocks)
        self._sectors = tuple(hamiltonians)
        self._dimensions = {
            sector: matrix.shape[0]
            for sector, matrix in hamiltonians.items()
        }

        raising_site = _as_sector_operator_blocks(
            raising_blocks,
            "raising_blocks",
            self._dimensions,
        )
        if lowering_blocks is None:
            lowering_site = {
                (source, target): block.conj().T
                for (target, source), block in raising_site.items()
            }
        else:
            lowering_site = _as_sector_operator_blocks(
                lowering_blocks,
                "lowering_blocks",
                self._dimensions,
            )

        self._energies = {}
        self._transforms = {}
        self._hamiltonian = {}
        for sector, matrix in hamiltonians.items():
            energies, transform = np.linalg.eigh(matrix)
            energies = np.real_if_close(energies).real
            self._energies[sector] = energies
            self._transforms[sector] = transform
            self._hamiltonian[sector] = np.diag(energies).astype(
                np.complex128
            )

        self._raising_flat = self._transform_blocks(raising_site)
        self._lowering_flat = self._transform_blocks(lowering_site)
        self._raising = _group_sector_operator_blocks(
            self._raising_flat, self._sectors
        )
        self._lowering = _group_sector_operator_blocks(
            self._lowering_flat, self._sectors
        )

        if detection_blocks is None:
            detection_site = self._combine_blocks(raising_site, lowering_site)
        else:
            detection_site = _as_sector_operator_blocks(
                detection_blocks,
                "detection_blocks",
                self._dimensions,
            )
        self._detection = _group_sector_operator_blocks(
            self._transform_blocks(detection_site), self._sectors
        )

        self._observable_inputs = {}
        if observable_op_blocks is not None:
            if not isinstance(observable_op_blocks, Mapping):
                raise TypeError("observable_op_blocks must be a mapping.")
            for name, blocks in observable_op_blocks.items():
                self._observable_inputs[str(name)] = (
                    _as_sector_operator_blocks(
                        blocks,
                        f"observable_op_blocks[{name!r}]",
                        self._dimensions,
                    )
                )
        self._observables = {
            name: _group_sector_operator_blocks(
                self._transform_blocks(blocks), self._sectors
            )
            for name, blocks in self._observable_inputs.items()
        }

        self._initial_sector = self._resolve_initial_sector(initial_sector)
        self._initial_state_index = int(initial_state_index)
        if not 0 <= self._initial_state_index < self.dimension(
            self._initial_sector
        ):
            raise ValueError(
                "initial_state_index is outside the selected initial sector."
            )

        self._channels = self._build_channels(c_ops_raw)

    @staticmethod
    def _combine_blocks(*block_mappings):
        combined = {}
        for blocks in block_mappings:
            for key, block in blocks.items():
                if key in combined:
                    combined[key] = combined[key] + block
                else:
                    combined[key] = block.copy()
        return combined

    def _transform_blocks(self, blocks):
        transformed = {}
        for (target, source), block in blocks.items():
            transformed[(target, source)] = (
                self._transforms[target].conj().T
                @ block
                @ self._transforms[source]
            )
        return transformed

    def _resolve_initial_sector(self, initial_sector):
        if initial_sector is not None:
            if initial_sector not in self._dimensions:
                raise KeyError(f"Unknown initial sector {initial_sector!r}.")
            return initial_sector
        return min(
            self._sectors,
            key=lambda sector: float(self._energies[sector][0]),
        )

    def _build_channels(self, c_ops_raw):
        channels = []
        for index, item in enumerate(c_ops_raw):
            if not (isinstance(item, tuple) and len(item) == 2):
                raise ValueError(
                    "c_ops_raw must contain (operator_blocks, gamma) pairs."
                )
            raw_blocks, gamma = item
            normalized = _as_sector_operator_blocks(
                raw_blocks,
                f"c_ops_raw[{index}]",
                self._dimensions,
            )
            transformed = _group_sector_operator_blocks(
                self._transform_blocks(normalized), self._sectors
            )
            channels.append(
                CollapseChannel(
                    name=f"c_op_{index}",
                    rate=float(gamma),
                    operator_blocks=transformed,
                )
            )
        return tuple(channels)

    # -- SectorModel contract ------------------------------------------------

    def sectors(self):
        return self._sectors

    def dimension(self, sector):
        return self._dimensions[sector]

    def hamiltonian_blocks(self, source):
        return {source: self._hamiltonian[source]}

    def transition_blocks(self, operator_name, direction, source):
        if direction == "plus":
            return dict(self._raising[source])
        if direction == "minus":
            return dict(self._lowering[source])
        raise ValueError("direction must be 'plus' or 'minus'.")

    def observable_blocks(self, observable_name, source):
        observable_name = str(observable_name)
        if observable_name in self._observables:
            return dict(self._observables[observable_name][source])
        if self._observables:
            available = ", ".join(sorted(self._observables))
            raise KeyError(
                f"Unknown observable {observable_name!r}; available: "
                f"{available}."
            )
        return dict(self._detection[source])

    def observable_names(self):
        if self._observables:
            return tuple(sorted(self._observables))
        return ("polarization",)

    def transition_decomposition(self):
        return "explicit_sector"

    def initial_condition(self, context=None):
        vector = np.zeros(
            self.dimension(self._initial_sector), dtype=np.complex128
        )
        vector[self._initial_state_index] = 1.0
        return PureState(
            sector=self._initial_sector,
            vector=vector,
            energy=float(
                self._energies[self._initial_sector][
                    self._initial_state_index
                ]
            ),
        )

    def equilibrium_state(self, context):
        temperature = float(context.temperature)
        minimum = min(
            float(energies[0]) for energies in self._energies.values()
        )
        populations = {}
        if temperature <= 0:
            degeneracy = sum(
                int(
                    np.count_nonzero(
                        np.isclose(
                            energies, minimum, atol=1e-12, rtol=0.0
                        )
                    )
                )
                for energies in self._energies.values()
            )
            for sector, energies in self._energies.items():
                populations[sector] = (
                    np.isclose(
                        energies, minimum, atol=1e-12, rtol=0.0
                    ).astype(float)
                    / degeneracy
                )
        else:
            denominator = self._boltzmann_constant * temperature
            unnormalized = {
                sector: np.exp(-(energies - minimum) / denominator)
                for sector, energies in self._energies.items()
            }
            partition = sum(values.sum() for values in unnormalized.values())
            populations = {
                sector: values / partition
                for sector, values in unnormalized.items()
            }
        return DensityState(
            blocks={
                (sector, sector): DenseDensityBlock(
                    np.diag(values.astype(np.complex128))
                )
                for sector, values in populations.items()
            }
        )

    def collapse_channels(self, context=None):
        return self._channels

    def requirements(self):
        return ModelRequirements(
            state_kind="pure",
            generator_kind="lindblad" if self._channels else "unitary",
            domains=("time", "frequency"),
            exactness="exact",
        )


class EigenbasisKModel:
    """Site-basis arrays -> diagonalized, sector-based k model.

    Optional arrays override transition splitting and k-point weights.
    ``boltzmann_constant`` must use the same energy unit as ``H_model``; its
    default value is in eV/K.
    """

    def __init__(
        self,
        H_model,
        interaction_op_array,
        *,
        c_ops_raw=(),
        detection_op_array=None,
        observable_op_arrays=None,
        j_plus_array=None,
        j_minus_array=None,
        rwa_tol=1e-6,
        k_weights=None,
        boltzmann_constant=_BOLTZMANN_EV_PER_KELVIN,
    ):
        self._boltzmann_constant = _validated_boltzmann_constant(
            boltzmann_constant
        )
        H_stack = _as_k_stack(H_model, "H_model")
        n_k, d, _ = H_stack.shape
        interaction_stack = _as_k_stack(
            interaction_op_array, "interaction_op_array", expected_d=d
        )
        if interaction_stack.shape[0] not in (1, n_k):
            raise ValueError(
                "interaction_op_array must have 1 or N_k slices."
            )
        if interaction_stack.shape[0] == 1 and n_k > 1:
            interaction_stack = np.repeat(interaction_stack, n_k, axis=0)

        explicit_plus = j_plus_array is not None
        explicit_minus = j_minus_array is not None
        if explicit_plus != explicit_minus:
            raise ValueError(
                "j_plus_array and j_minus_array must be provided together."
            )

        if explicit_plus:
            j_plus_stack = _as_k_stack(
                j_plus_array, "j_plus_array", expected_d=d
            )
            j_minus_stack = _as_k_stack(
                j_minus_array, "j_minus_array", expected_d=d
            )
            for name, stack in (
                ("j_plus_array", j_plus_stack),
                ("j_minus_array", j_minus_stack),
            ):
                if stack.shape[0] not in (1, n_k):
                    raise ValueError(
                        f"{name} must have 1 or N_k slices."
                    )
            if j_plus_stack.shape[0] == 1 and n_k > 1:
                j_plus_stack = np.repeat(j_plus_stack, n_k, axis=0)
            if j_minus_stack.shape[0] == 1 and n_k > 1:
                j_minus_stack = np.repeat(j_minus_stack, n_k, axis=0)

        detection_stack = (
            interaction_stack
            if detection_op_array is None
            else _as_k_stack(detection_op_array, "detection_op_array", expected_d=d)
        )
        if detection_stack.shape[0] == 1 and n_k > 1:
            detection_stack = np.repeat(detection_stack, n_k, axis=0)

        weights = (
            np.ones(n_k, dtype=float) / n_k
            if k_weights is None
            else np.asarray(k_weights, dtype=float)
        )
        if weights.shape != (n_k,):
            raise ValueError("k_weights must contain N_k values.")
        if np.any(weights < 0):
            raise ValueError("k_weights must be non-negative.")
        weights = weights / weights.sum()

        self._n_k = n_k
        self._d = d
        self._rwa_tol = float(rwa_tol)
        self._transition_decomposition = (
            "explicit" if explicit_plus else "automatic_energy"
        )
        self._sectors = tuple(f"k{index}" for index in range(n_k))
        self._weights = dict(zip(self._sectors, weights))

        self._energies = {}
        self._hamiltonian = {}
        self._J_plus = {}
        self._J_minus = {}
        self._detection = {}
        self._observable_arrays = {}
        if observable_op_arrays is not None:
            if not isinstance(observable_op_arrays, Mapping):
                raise TypeError("observable_op_arrays must be a mapping.")
            for name, array in observable_op_arrays.items():
                stack = _as_k_stack(
                    array,
                    f"observable_op_arrays[{name!r}]",
                    expected_d=d,
                )
                if stack.shape[0] not in (1, n_k):
                    raise ValueError(
                        f"observable_op_arrays[{name!r}] must have 1 or N_k slices."
                    )
                if stack.shape[0] == 1 and n_k > 1:
                    stack = np.repeat(stack, n_k, axis=0)
                self._observable_arrays[str(name)] = stack

        self._observables = {name: {} for name in self._observable_arrays}
        for index, sector in enumerate(self._sectors):
            evals, U = np.linalg.eigh(H_stack[index])
            evals = np.real_if_close(evals).real
            self._energies[sector] = evals
            self._hamiltonian[sector] = np.diag(evals).astype(np.complex128)

            if explicit_plus:
                self._J_plus[sector] = (
                    U.conj().T @ j_plus_stack[index] @ U
                )
                self._J_minus[sector] = (
                    U.conj().T @ j_minus_stack[index] @ U
                )
            else:
                interaction_eigen = U.conj().T @ interaction_stack[index] @ U
                delta_E = evals[:, np.newaxis] - evals[np.newaxis, :]
                self._J_plus[sector] = np.where(
                    delta_E > self._rwa_tol, interaction_eigen, 0.0
                )
                self._J_minus[sector] = np.where(
                    delta_E < -self._rwa_tol, interaction_eigen, 0.0
                )
            self._detection[sector] = U.conj().T @ detection_stack[index] @ U
            for name, stack in self._observable_arrays.items():
                self._observables[name][sector] = (
                    U.conj().T @ stack[index] @ U
                )

        self._channels = self._build_channels(c_ops_raw, H_stack)

    def _build_channels(self, c_ops_raw, H_stack):
        channels = []
        for index, item in enumerate(c_ops_raw):
            if not (isinstance(item, tuple) and len(item) == 2):
                raise ValueError(
                    "c_ops_raw must contain (operator, gamma) pairs."
                )
            operator_raw, gamma = item
            operator_stack = _as_k_stack(
                operator_raw, f"c_ops_raw[{index}]", expected_d=self._d
            )
            if operator_stack.shape[0] == 1 and self._n_k > 1:
                operator_stack = np.repeat(operator_stack, self._n_k, axis=0)

            operator_blocks = {}
            for k_index, sector in enumerate(self._sectors):
                evals, U = np.linalg.eigh(H_stack[k_index])
                operator_eigen = U.conj().T @ operator_stack[k_index] @ U
                operator_blocks[sector] = {sector: operator_eigen}

            channels.append(
                CollapseChannel(
                    name=f"c_op_{index}",
                    rate=float(gamma),
                    operator_blocks=operator_blocks,
                )
            )
        return tuple(channels)

    # -- SectorModel contract ------------------------------------------------

    def sectors(self):
        return self._sectors

    def dimension(self, sector):
        return self._d

    def hamiltonian_blocks(self, source):
        return {source: self._hamiltonian[source]}

    def transition_blocks(self, operator_name, direction, source):
        operator = (
            self._J_plus[source] if direction == "plus" else self._J_minus[source]
        )
        return {source: operator}

    def observable_blocks(self, observable_name, source):
        observable_name = str(observable_name)
        if observable_name in self._observables:
            return {source: self._observables[observable_name][source]}
        if self._observable_arrays:
            available = ", ".join(sorted(self._observables))
            raise KeyError(
                f"Unknown observable {observable_name!r}; available: {available}."
            )
        # Legacy API: one detection operator could serve any pathway name.
        return {source: self._detection[source]}

    def observable_names(self):
        """Return observable names explicitly provided to the adapter."""
        if self._observable_arrays:
            return tuple(sorted(self._observable_arrays))
        return ("polarization",)

    def transition_decomposition(self):
        """Return whether J+ / J- are inferred or explicitly provided."""
        return self._transition_decomposition

    def initial_condition(self, context=None):
        blocks = {}
        for sector in self._sectors:
            population = np.zeros((self._d, self._d), dtype=np.complex128)
            population[0, 0] = self._weights[sector]
            blocks[(sector, sector)] = DenseDensityBlock(population)
        return DensityState(blocks=blocks)

    def equilibrium_state(self, context):
        temperature = float(context.temperature)
        blocks = {}
        for sector in self._sectors:
            evals = self._energies[sector]
            if temperature <= 0:
                populations = np.zeros(self._d, dtype=float)
                populations[0] = 1.0
            else:
                shifted = evals - np.min(evals)
                boltzmann = np.exp(
                    -shifted / (self._boltzmann_constant * temperature)
                )
                populations = boltzmann / boltzmann.sum()
            matrix = np.diag(
                (self._weights[sector] * populations).astype(np.complex128)
            )
            blocks[(sector, sector)] = DenseDensityBlock(matrix)
        return DensityState(blocks=blocks)

    def collapse_channels(self, context=None):
        return self._channels

    def requirements(self):
        return ModelRequirements(
            state_kind="mixed",
            generator_kind="lindblad" if self._channels else "unitary",
            domains=("time", "frequency"),
            exactness="exact",
        )
