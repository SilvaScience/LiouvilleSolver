"""Matrix-free Liouville backend with low-rank density operators.

Intermediate states are stored as factorized ``X @ S @ Y.conj().T`` objects.
Time propagation preserves this representation, while restarted GMRES solves
frequency-domain systems with SVD recompression. The method trades controlled
approximation for reduced memory use on suitable problems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, expm_multiply, gmres

from ..backends.base import BackendBase
from ..capabilities import BackendCapabilities, Capabilities
from ..exceptions import CapabilityError, ConvergenceError
from ..results import PathwayResult


@dataclass(frozen=True)
class _LowRankState:
    """Operator represented as ``X @ S @ Y.conj().T``."""

    X: np.ndarray
    S: np.ndarray
    Y: np.ndarray

    def __post_init__(self):
        X = np.asarray(self.X, dtype=np.complex128)
        S = np.asarray(self.S, dtype=np.complex128)
        Y = np.asarray(self.Y, dtype=np.complex128)
        if X.ndim != 2 or Y.ndim != 2 or S.ndim != 2:
            raise ValueError("Low-rank factors must be matrices.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must act on the same Hilbert space.")
        if S.shape != (X.shape[1], Y.shape[1]):
            raise ValueError(
                "S must have shape (X_columns, Y_columns)."
            )
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "S", S)
        object.__setattr__(self, "Y", Y)

    @classmethod
    def zero(cls, dimension):
        empty = np.zeros((int(dimension), 0), dtype=np.complex128)
        return cls(empty, np.zeros((0, 0), dtype=np.complex128), empty.copy())

    @classmethod
    def rank_one(cls, ket, bra=None):
        ket = np.asarray(ket, dtype=np.complex128).reshape(-1, 1)
        bra = ket if bra is None else np.asarray(
            bra, dtype=np.complex128
        ).reshape(-1, 1)
        return cls(ket, np.ones((1, 1), dtype=np.complex128), bra)

    @classmethod
    def from_dense(cls, matrix):
        matrix = np.asarray(matrix, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Density matrix must be square.")
        U, singular_values, Vh = np.linalg.svd(
            matrix, full_matrices=False
        )
        if singular_values.size == 0 or singular_values[0] == 0:
            return cls.zero(matrix.shape[0])
        return cls(
            U,
            np.diag(singular_values.astype(np.complex128)),
            Vh.conj().T,
        )

    @property
    def dimension(self):
        return self.X.shape[0]

    @property
    def rank(self):
        return min(self.X.shape[1], self.Y.shape[1])

    @property
    def is_zero(self):
        return self.S.size == 0 or not np.any(self.S)

    def scaled(self, coefficient):
        if self.is_zero or coefficient == 0:
            return self.zero(self.dimension)
        return _LowRankState(self.X, complex(coefficient) * self.S, self.Y)

    def inner(self, other):
        """Other low-rank state -> matrix-free Frobenius product."""
        if self.is_zero or other.is_zero:
            return 0.0j
        x_overlap = self.X.conj().T @ other.X
        y_overlap = other.Y.conj().T @ self.Y
        return np.trace(
            self.S.conj().T @ x_overlap @ other.S @ y_overlap
        )

    def norm(self):
        value = float(np.real(self.inner(self)))
        return float(np.sqrt(max(value, 0.0)))

    @classmethod
    def linear_combination(cls, terms, dimension):
        """Weighted low-rank terms -> factorized sum."""
        active = [
            (complex(coefficient), state)
            for coefficient, state in terms
            if coefficient != 0 and not state.is_zero
        ]
        if not active:
            return cls.zero(dimension)

        x_columns = sum(state.X.shape[1] for _, state in active)
        y_columns = sum(state.Y.shape[1] for _, state in active)
        X = np.concatenate([state.X for _, state in active], axis=1)
        Y = np.concatenate([state.Y for _, state in active], axis=1)
        S = np.zeros((x_columns, y_columns), dtype=np.complex128)
        x_offset = 0
        y_offset = 0
        for coefficient, state in active:
            x_size, y_size = state.S.shape
            S[
                x_offset : x_offset + x_size,
                y_offset : y_offset + y_size,
            ] = coefficient * state.S
            x_offset += x_size
            y_offset += y_size
        return cls(X, S, Y)

    def compressed(self, *, relative_tolerance, absolute_tolerance, max_rank):
        """Low-rank factors -> QR/SVD-compressed state."""
        if self.is_zero:
            return self.zero(self.dimension), 0.0

        Qx, Rx = np.linalg.qr(self.X, mode="reduced")
        Qy, Ry = np.linalg.qr(self.Y, mode="reduced")
        core = Rx @ self.S @ Ry.conj().T
        U, singular_values, Vh = np.linalg.svd(core, full_matrices=False)
        if singular_values.size == 0 or singular_values[0] == 0:
            return self.zero(self.dimension), 0.0

        threshold = max(
            float(absolute_tolerance),
            float(relative_tolerance) * float(singular_values[0]),
        )
        retained = int(np.count_nonzero(singular_values > threshold))
        retained = max(1, min(retained, int(max_rank)))
        discarded = float(np.linalg.norm(singular_values[retained:]))

        X = Qx @ U[:, :retained]
        Y = Qy @ Vh.conj().T[:, :retained]
        S = np.diag(singular_values[:retained].astype(np.complex128))
        return _LowRankState(X, S, Y), discarded


class LowRankLiouvilleBackend(BackendBase):
    """Pathway inputs -> low-rank Liouville propagation and resolvents.

    Restarted GMRES recompresses Krylov vectors with SVD.
    """

    name = "low_rank_liouville"
    capabilities = Capabilities(
        pure_state=True,
        mixed_state=True,
        time_domain=True,
        frequency_domain=True,
        hilbert_resolvent=True,
        lindblad=True,
        finite_temperature=True,
        coupled_sectors=True,
        matrix_free=True,
        low_rank=True,
        metadata=(
            ("lindblad_time_domain", False),
            ("lindblad_frequency_domain", True),
        ),
    )
    backend_capabilities = BackendCapabilities(
        state_kinds=("pure", "mixed"),
        supported_pairs=(
            ("unitary", "time"),
            ("unitary", "frequency"),
            ("lindblad", "frequency"),
        ),
        exactness="truncated_low_rank",
        matrix_free=True,
        low_rank=True,
    )

    def __init__(
        self,
        *,
        eta=0.05,
        low_rank_tolerance=1e-10,
        low_rank_absolute_tolerance=0.0,
        max_rank=64,
        krylov_tolerance=1e-9,
        krylov_maxiter=100,
        resolvent_restart=20,
        **options,
    ):
        super().__init__(eta=eta, **options)
        self.low_rank_tolerance = float(low_rank_tolerance)
        self.low_rank_absolute_tolerance = float(
            low_rank_absolute_tolerance
        )
        self.max_rank = int(max_rank)
        self.krylov_tolerance = float(krylov_tolerance)
        self.krylov_maxiter = int(krylov_maxiter)
        self.resolvent_restart = int(resolvent_restart)
        if self.low_rank_tolerance < 0:
            raise ValueError("low_rank_tolerance must be non-negative.")
        if self.low_rank_absolute_tolerance < 0:
            raise ValueError(
                "low_rank_absolute_tolerance must be non-negative."
            )
        if self.max_rank < 1:
            raise ValueError("max_rank must be positive.")
        if self.krylov_tolerance <= 0:
            raise ValueError("krylov_tolerance must be positive.")
        if self.krylov_maxiter < 1 or self.resolvent_restart < 1:
            raise ValueError(
                "krylov_maxiter and resolvent_restart must be positive."
            )
        self._initial_low_rank_state = None
        self._initial_compression_discarded = 0.0
        self._discarded_norm_squared = 0.0
        self._maximum_observed_rank = 0

    def build(self, model, context=None):
        super().build(model, context=context)
        state = _LowRankState.from_dense(self._initial_density_matrix)
        (
            self._initial_low_rank_state,
            self._initial_compression_discarded,
        ) = state.compressed(
            relative_tolerance=self.low_rank_tolerance,
            absolute_tolerance=self.low_rank_absolute_tolerance,
            max_rank=self.max_rank,
        )
        return self

    def clear_caches(self):
        """Return no materialized propagator or resolvent."""

    def _reset_diagnostics(self):
        self._discarded_norm_squared = 0.0
        self._maximum_observed_rank = 0

    def _compress(self, state):
        compressed, discarded = state.compressed(
            relative_tolerance=self.low_rank_tolerance,
            absolute_tolerance=self.low_rank_absolute_tolerance,
            max_rank=self.max_rank,
        )
        self._discarded_norm_squared += discarded * discarded
        self._maximum_observed_rank = max(
            self._maximum_observed_rank, compressed.rank
        )
        return compressed

    def _combine(self, terms):
        state = _LowRankState.linear_combination(
            terms, self.layout.total_dimension
        )
        return self._compress(state)

    def _apply_to_factor_columns(self, factors, action):
        if factors.shape[1] == 0:
            return factors.copy()
        result = np.empty_like(factors)
        for column in range(factors.shape[1]):
            result[:, column] = action(factors[:, column])
        return result

    def _generator_action(self, state):
        """Low-rank density state -> generator action ``A rho``."""
        if state.is_zero:
            return state
        h_x = self._apply_to_factor_columns(
            state.X, self._hamiltonian.matvec
        )
        h_y = self._apply_to_factor_columns(
            state.Y, self._hamiltonian.rmatvec
        )
        left = _LowRankState(h_x, state.S, state.Y)
        right = _LowRankState(state.X, state.S, h_y)
        terms = [(-1j, left), (1j, right)]

        for channel, operator in self._collapse_operators:
            c_x = self._apply_to_factor_columns(
                state.X, operator.matvec
            )
            c_y = self._apply_to_factor_columns(
                state.Y, operator.matvec
            )
            k_x = self._apply_to_factor_columns(
                state.X,
                lambda value, op=operator: op.rmatvec(op.matvec(value)),
            )
            k_y = self._apply_to_factor_columns(
                state.Y,
                lambda value, op=operator: op.rmatvec(op.matvec(value)),
            )
            terms.extend(
                (
                    (
                        channel.rate,
                        _LowRankState(c_x, state.S, c_y),
                    ),
                    (
                        -0.5 * channel.rate,
                        _LowRankState(k_x, state.S, state.Y),
                    ),
                    (
                        -0.5 * channel.rate,
                        _LowRankState(state.X, state.S, k_y),
                    ),
                )
            )
        return self._combine(terms)

    def _shifted_generator_action(self, state, omega, eta):
        shift = self.generator.frequency_shift(omega, eta)
        generator_state = self._generator_action(state)
        return self._combine(
            ((shift, state), (-1.0, generator_state))
        )

    def _apply_interaction(self, state, interaction):
        if state.is_zero:
            return state
        if interaction.side == "ket":
            X = self._apply_to_factor_columns(
                state.X,
                lambda vector: self.apply_transition(
                    vector,
                    operator_name=interaction.operator,
                    direction=interaction.direction,
                    source_sector=interaction.source_sector,
                    target_sector=interaction.target_sector,
                ),
            )
            return self._compress(_LowRankState(X, state.S, state.Y))

        Y = self._apply_to_factor_columns(
            state.Y,
            lambda vector: self.apply_transition(
                vector,
                operator_name=interaction.operator,
                direction=interaction.direction,
                source_sector=interaction.source_sector,
                target_sector=interaction.target_sector,
            ),
        )
        return self._compress(_LowRankState(state.X, state.S, Y))

    def _propagate(self, state, delay):
        delay = float(delay)
        if state.is_zero or delay == 0:
            return state
        if not self.generator.is_unitary:
            raise CapabilityError(
                "LowRankLiouvilleBackend supports Lindblad terms in the direct "
                "resolvent but not yet in time propagation. Use "
                "dense_liouville or sparse_sector for dissipative intervals."
            )
        generator = (-1j * delay) * self._hamiltonian
        X = np.asarray(
            expm_multiply(generator, state.X, traceA=0.0),
            dtype=np.complex128,
        )
        Y = np.asarray(
            expm_multiply(generator, state.Y, traceA=0.0),
            dtype=np.complex128,
        )
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        return self._compress(_LowRankState(X, state.S, Y))

    def _solve_low_rank_resolvent(self, rhs, omega, eta):
        """Low-rank source and shift -> restarted GMRES solution."""
        rhs_norm = rhs.norm()
        if rhs_norm == 0:
            return rhs, {
                "iterations": 0,
                "residual_norm": 0.0,
                "relative_residual": 0.0,
                "rank_history": (0,),
            }

        target = self.krylov_tolerance * rhs_norm
        solution = _LowRankState.zero(self.layout.total_dimension)
        iterations = 0
        rank_history = [rhs.rank]
        residual_norm = rhs_norm

        while iterations < self.krylov_maxiter:
            action_solution = self._shifted_generator_action(
                solution, omega, eta
            )
            residual = self._combine(
                ((1.0, rhs), (-1.0, action_solution))
            )
            beta = residual.norm()
            residual_norm = beta
            if beta <= target:
                break

            cycle_size = min(
                self.resolvent_restart,
                self.krylov_maxiter - iterations,
            )
            basis = [residual.scaled(1.0 / beta)]
            hessenberg = np.zeros(
                (cycle_size + 1, cycle_size), dtype=np.complex128
            )
            right_hand_side = np.zeros(
                cycle_size + 1, dtype=np.complex128
            )
            right_hand_side[0] = beta
            cycle_origin = solution
            candidate = solution
            breakdown = False

            for column in range(cycle_size):
                vector = self._shifted_generator_action(
                    basis[column], omega, eta
                )
                for row in range(column + 1):
                    coefficient = basis[row].inner(vector)
                    hessenberg[row, column] += coefficient
                    vector = self._combine(
                        ((1.0, vector), (-coefficient, basis[row]))
                    )

                # A second pass limits orthogonality loss from recompression.
                for row in range(column + 1):
                    correction = basis[row].inner(vector)
                    hessenberg[row, column] += correction
                    vector = self._combine(
                        ((1.0, vector), (-correction, basis[row]))
                    )

                next_norm = vector.norm()
                hessenberg[column + 1, column] = next_norm
                if next_norm > np.finfo(float).eps:
                    basis.append(vector.scaled(1.0 / next_norm))
                else:
                    breakdown = True

                coefficients, *_ = np.linalg.lstsq(
                    hessenberg[: column + 2, : column + 1],
                    right_hand_side[: column + 2],
                    rcond=None,
                )
                terms = [(1.0, cycle_origin)]
                terms.extend(
                    (coefficients[index], basis[index])
                    for index in range(column + 1)
                )
                candidate = self._combine(terms)
                true_residual = self._combine(
                    (
                        (1.0, rhs),
                        (
                            -1.0,
                            self._shifted_generator_action(
                                candidate, omega, eta
                            ),
                        ),
                    )
                )
                residual_norm = true_residual.norm()
                iterations += 1
                rank_history.append(candidate.rank)
                if residual_norm <= target or breakdown:
                    break

            solution = candidate
            if residual_norm <= target:
                break
            if breakdown:
                break

        diagnostics = {
            "iterations": iterations,
            "residual_norm": residual_norm,
            "relative_residual": residual_norm / rhs_norm,
            "rank_history": tuple(rank_history),
        }
        if residual_norm > target:
            raise ConvergenceError(
                "Low-rank resolvent did not converge "
                f"(iterations={iterations}, relative residual="
                f"{residual_norm / rhs_norm:.3e}, rank={solution.rank})."
            )
        return solution.scaled(
            self.generator.frequency_prefactor
        ), diagnostics

    def _detect(self, state, observable_name):
        if state.is_zero:
            return 0.0j
        observable_x = self._apply_to_factor_columns(
            state.X,
            lambda vector: self.apply_observable(vector, observable_name),
        )
        return np.trace(
            state.Y.conj().T @ observable_x @ state.S
        )

    def solve_hilbert_resolvent(
        self,
        vector,
        omega,
        *,
        eta=None,
        initial_guess=None,
    ):
        """Vector and frequency -> Hilbert resolvent independent of rho rank."""
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        if vector.size != self.layout.total_dimension:
            raise ValueError("Size is incompatible with the Hilbert space.")
        eta = self.eta if eta is None else float(eta)
        shift = complex(float(omega), eta)
        dimension = self.layout.total_dimension
        operator = LinearOperator(
            (dimension, dimension),
            dtype=np.complex128,
            matvec=lambda value: shift * value
            - self._hamiltonian.matvec(value),
            rmatvec=lambda value: np.conj(shift) * value
            - self._hamiltonian.rmatvec(value),
        )
        solution, info = gmres(
            operator,
            vector,
            x0=initial_guess,
            rtol=self.krylov_tolerance,
            atol=0.0,
            restart=self.resolvent_restart,
            maxiter=self.krylov_maxiter,
        )
        residual = float(np.linalg.norm(operator @ solution - vector))
        if info != 0:
            raise ConvergenceError(
                "Hilbert GMRES resolvent did not converge "
                f"(info={info}, residual={residual:.3e})."
            )
        return np.asarray(solution, dtype=np.complex128), {
            "gmres_info": int(info),
            "residual_norm": residual,
            "omega": float(omega),
            "eta": eta,
        }

    def calc_pathway(self, pathway, protocol, coordinates):
        """Pathway inputs -> response computed entirely as ``X S Y†``."""
        protocol.validate_pathway(pathway)
        protocol.validate_coordinates(coordinates)
        self._reset_diagnostics()
        response = self._initial_low_rank_state
        resolvent_diagnostics = []

        for interaction, interval in zip(
            pathway.interactions, protocol.intervals
        ):
            response = self._apply_interaction(response, interaction)
            if interval.domain == "time":
                response = self._propagate(
                    response, coordinates[interval.name]
                )
            elif interval.domain == "frequency":
                eta = self.eta if interval.eta is None else interval.eta
                response, diagnostics = self._solve_low_rank_resolvent(
                    response,
                    coordinates[interval.name],
                    eta,
                )
                resolvent_diagnostics.append(
                    {"interval": interval.name, **diagnostics}
                )

        value = (
            pathway.response_prefactor
            * self._detect(response, pathway.detection)
        )
        return PathwayResult(
            value=complex(value),
            pathway=pathway.name,
            diagnostics={
                "backend": self.name,
                "hilbert_dimension": self.layout.total_dimension,
                "representation": "low_rank_liouville",
                "generator_kind": self.generator.kind,
                "generator_convention": "A=-i[H,.]+D",
                "final_rank": response.rank,
                "maximum_observed_rank": self._maximum_observed_rank,
                "compression_discarded_norm": float(
                    np.sqrt(self._discarded_norm_squared)
                ),
                "initial_compression_discarded_norm": (
                    self._initial_compression_discarded
                ),
                "resolvents": tuple(resolvent_diagnostics),
            },
        )
