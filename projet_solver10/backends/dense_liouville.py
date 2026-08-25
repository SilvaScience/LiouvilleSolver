"""Dense Liouville-space backend for small reference systems.

The backend materializes the full Liouvillian and evaluates propagation,
resolvents, pathways, and observables with dense linear algebra. It provides
an exact baseline for validation and compact models where the squared Hilbert
dimension remains affordable.
"""

from collections import OrderedDict

import numpy as np
from scipy.linalg import expm

from .base import BackendBase
from ..capabilities import BackendCapabilities, Capabilities
from ..observables import ObservableSpec
from ..results import PathwayResult


class DenseLiouvilleBackend(BackendBase):
    """Exact backend that materializes the full Liouville space."""

    name = "dense_liouville"
    capabilities = Capabilities(
        pure_state=True,
        mixed_state=True,
        time_domain=True,
        frequency_domain=True,
        hilbert_resolvent=True,
        lindblad=True,
        finite_temperature=True,
        coupled_sectors=True,
        matrix_free=False,
    )
    backend_capabilities = BackendCapabilities(
        state_kinds=("pure", "mixed"),
        supported_pairs=(
            ("unitary", "time"),
            ("unitary", "frequency"),
            ("lindblad", "time"),
            ("lindblad", "frequency"),
        ),
        exactness="exact",
        matrix_free=False,
    )

    def __init__(
        self,
        *,
        eta=0.05,
        cache_resolvents=True,
        max_cache_entries=32,
        **options,
    ):
        super().__init__(eta=eta, **options)
        self.cache_resolvents = bool(cache_resolvents)
        self.max_cache_entries = int(max_cache_entries)
        self._resolvent_cache = OrderedDict()
        self._time_cache = OrderedDict()
        self._H_dense = None
        self._A_dense = None
        self._I_liouville = None
        self._rho_initial = None
        self._transition_matrix_cache = {}
        self._observable_matrix_cache = {}

    def build(self, model, context=None):
        super().build(model, context=context)
        dimension = self.layout.total_dimension
        self._H_dense = self.materialize(self._hamiltonian)
        self._A_dense = self.materialize(self.generator.linear_operator)
        self._I_liouville = np.eye(
            dimension * dimension, dtype=np.complex128
        )
        self._rho_initial = self._initial_density_matrix.reshape(
            -1, order="F"
        )
        return self

    def clear_caches(self):
        self._resolvent_cache.clear()
        self._time_cache.clear()

    def _bounded_cache_put(self, cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.max_cache_entries:
            cache.popitem(last=False)

    def _resolvent(self, omega, eta):
        key = (round(float(omega), 14), round(float(eta), 14))
        cached = self._resolvent_cache.get(key)
        if cached is not None:
            self._resolvent_cache.move_to_end(key)
            return cached
        shift = self.generator.frequency_shift(omega, eta)
        matrix = shift * self._I_liouville - self._A_dense
        inverse = self.generator.frequency_prefactor * np.linalg.inv(matrix)
        if self.cache_resolvents:
            self._bounded_cache_put(self._resolvent_cache, key, inverse)
        return inverse

    def _time_propagator(self, delay):
        key = round(float(delay), 14)
        cached = self._time_cache.get(key)
        if cached is not None:
            self._time_cache.move_to_end(key)
            return cached
        propagator = expm(self._A_dense * float(delay))
        self._bounded_cache_put(self._time_cache, key, propagator)
        return propagator

    def solve_hilbert_resolvent(self, vector, omega, *, eta=None, **_):
        """Vector and frequency -> exact dense Hilbert resolvent."""
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        dimension = self.layout.total_dimension
        if vector.size != dimension:
            raise ValueError("Size is incompatible with the Hilbert space.")
        eta = self.eta if eta is None else float(eta)
        matrix = (
            complex(float(omega), eta)
            * np.eye(dimension, dtype=np.complex128)
            - self._H_dense
        )
        solution = np.linalg.solve(matrix, vector)
        residual = float(np.linalg.norm(matrix @ solution - vector))
        return solution, {
            "method": "dense_solve",
            "residual_norm": residual,
            "omega": float(omega),
            "eta": eta,
        }

    def _transition_matrix(self, interaction):
        key = (
            interaction.operator,
            interaction.direction,
            interaction.source_sector,
            interaction.target_sector,
        )
        if key in self._transition_matrix_cache:
            return self._transition_matrix_cache[key]
        dimension = self.layout.total_dimension
        matrix = np.empty((dimension, dimension), dtype=np.complex128)
        basis = np.zeros(dimension, dtype=np.complex128)
        for index in range(dimension):
            basis[index] = 1.0
            matrix[:, index] = self.apply_transition(
                basis,
                operator_name=interaction.operator,
                direction=interaction.direction,
                source_sector=interaction.source_sector,
                target_sector=interaction.target_sector,
            )
            basis[index] = 0.0
        self._transition_matrix_cache[key] = matrix
        return matrix

    def _observable_matrix(self, observable):
        spec = (
            observable
            if isinstance(observable, ObservableSpec)
            else ObservableSpec(name=str(observable))
        )
        if spec.kind == "integrated_jump":
            raise ValueError(
                "An integrated_jump observable has no final matrix; "
                "use time-domain evaluation."
            )
        key = (spec.kind, spec.operator, spec.channel)
        if key in self._observable_matrix_cache:
            return self._observable_matrix_cache[key]
        dimension = self.layout.total_dimension
        matrix = np.empty((dimension, dimension), dtype=np.complex128)
        basis = np.zeros(dimension, dtype=np.complex128)
        if spec.kind == "jump_rate":
            for index in range(dimension):
                basis[index] = 1.0
                matrix[:, index] = self.apply_jump_effect(
                    basis, spec.channel
                )
                basis[index] = 0.0
        else:
            for index in range(dimension):
                basis[index] = 1.0
                matrix[:, index] = self.apply_observable(
                    basis, spec.operator
                )
                basis[index] = 0.0
        self._observable_matrix_cache[key] = matrix
        return matrix

    def _interaction_superoperator(self, interaction):
        operator = self._transition_matrix(interaction)
        identity = np.eye(self.layout.total_dimension, dtype=np.complex128)
        if interaction.label in {"Ku", "Kd"}:
            return np.kron(identity, operator)

        # For bra interactions, direction describes the operator acting on the
        # bra vector; right multiplication of rho therefore uses its adjoint.
        right_operator = operator.conj().T
        return np.kron(right_operator.T, identity)

    def _propagate_pathway(self, pathway, protocol, coordinates):
        protocol.validate_pathway(pathway)
        protocol.validate_coordinates(coordinates)
        response = self._rho_initial.copy()
        for interaction, interval in zip(
            pathway.interactions, protocol.intervals
        ):
            response = self._interaction_superoperator(interaction) @ response
            if interval.domain == "time":
                response = self._time_propagator(
                    coordinates[interval.name]
                ) @ response
            elif interval.domain == "frequency":
                eta = self.eta if interval.eta is None else interval.eta
                response = self._resolvent(
                    coordinates[interval.name], eta
                ) @ response

        return response

    def _detect_response(self, response, observable):
        dimension = self.layout.total_dimension
        rho = response.reshape((dimension, dimension), order="F")
        matrix = self._observable_matrix(observable)
        value = np.trace(matrix @ rho)
        if observable.kind == "jump_rate":
            value *= observable.efficiency
        return value

    def _projection_branches(self, response, observable):
        projections = observable.projection_interactions
        if not projections:
            return ((response, 1.0 + 0.0j),)
        return tuple(
            (
                self._interaction_superoperator(interaction) @ response,
                interaction.perturbative_prefactor,
            )
            for interaction in projections
        )

    def _detect_observable(self, response, observable):
        detector = observable.without_projection()
        return sum(
            prefactor * self._detect_response(projected, detector)
            for projected, prefactor in self._projection_branches(
                response, observable
            )
        )

    def _integrated_jump_values(self, response, observables):
        values = {}
        for spec in observables:
            start, stop = spec.time_window
            times = np.linspace(start, stop, spec.n_steps)
            detector = spec.instantaneous().without_projection()
            samples = []
            for time in times:
                value = 0.0j
                for projected, prefactor in self._projection_branches(
                    response, spec
                ):
                    propagated = self._time_propagator(time) @ projected
                    value += prefactor * self._detect_response(
                        propagated, detector
                    )
                samples.append(value)
            values[spec.name] = np.trapezoid(
                np.asarray(samples, dtype=np.complex128), times
            )
        return values

    def calc_pathway_observables(
        self, pathway, protocol, coordinates, observables
    ):
        """One pathway propagation -> responses for multiple observables."""
        response = self._propagate_pathway(pathway, protocol, coordinates)
        specs = tuple(observables)
        values = {}
        integrated = []
        for spec in specs:
            if spec.kind == "integrated_jump":
                integrated.append(spec)
            else:
                values[spec.name] = self._detect_observable(response, spec)
        values.update(self._integrated_jump_values(response, integrated))
        values = {
            name: pathway.response_prefactor * value
            for name, value in values.items()
        }
        diagnostics = {
            "backend": self.name,
            "hilbert_dimension": self.layout.total_dimension,
            "liouville_dimension": self.layout.total_dimension**2,
            "representation": "dense_liouville",
            "generator_kind": self.generator.kind,
            "generator_convention": "A=-i[H,.]+D",
            "observable_names": tuple(spec.name for spec in specs),
        }
        return PathwayResult(
            value=complex(values[pathway.detection]),
            pathway=pathway.name,
            diagnostics=diagnostics,
            observables={name: complex(value) for name, value in values.items()},
        )

    def calc_pathway(self, pathway, protocol, coordinates):
        """Pathway inputs -> response for its default observable."""
        result = self.calc_pathway_observables(
            pathway,
            protocol,
            coordinates,
            (ObservableSpec(name=pathway.detection),),
        )
        return result
