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

from .contracts import CollapseChannel, DensityState, DenseDensityBlock
from .capabilities import ModelRequirements


_BOLTZMANN_EV_PER_KELVIN = 8.617333262e-5


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


class EigenbasisKModel:
    """Site-basis arrays -> diagonalized, sector-based k model.

    Optional arrays override transition splitting and k-point weights.
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
    ):
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
                    -shifted / (_BOLTZMANN_EV_PER_KELVIN * temperature)
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
