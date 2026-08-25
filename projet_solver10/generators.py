"""Shared generator for time-domain Liouville-space dynamics.

``EvolutionGenerator`` combines a Hamiltonian with optional GKSL collapse
channels and exposes matrix-free forward and adjoint actions. Backends use the
same object for time propagation and shifted frequency-domain resolvents,
which keeps sign and damping conventions consistent.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator


class EvolutionGenerator:
    """Hamiltonian and collapse operators -> matrix-free ``A rho``.

    Frequency intervals apply ``[(eta-i*omega)I - A]**-1``.
    """

    # Exclude the legacy Green-function ``-1j`` factor to avoid rotating the
    # pathway perturbative coefficient twice.
    frequency_prefactor = 1.0

    def __init__(self, hamiltonian, collapse_operators=()):
        self.hamiltonian = hamiltonian
        self.collapse_operators = tuple(collapse_operators)
        self.dimension = int(hamiltonian.shape[0])
        self.liouville_dimension = self.dimension * self.dimension
        self.linear_operator = LinearOperator(
            (self.liouville_dimension, self.liouville_dimension),
            dtype=np.complex128,
            matvec=self.matvec,
            rmatvec=self.rmatvec,
        )

    @property
    def is_unitary(self):
        return not self.collapse_operators

    @property
    def kind(self):
        return "unitary" if self.is_unitary else "lindblad"

    def _as_matrix(self, value):
        return np.asarray(value, dtype=np.complex128).reshape(
            (self.dimension, self.dimension), order="F"
        )

    def _apply_columns(self, matrix, action):
        result = np.empty_like(matrix)
        for column in range(self.dimension):
            result[:, column] = action(matrix[:, column])
        return result

    def _left(self, matrix, action):
        return self._apply_columns(matrix, action)

    def _right(self, matrix, adjoint_action):
        """Adjoint action of O -> right multiplication by O."""
        acted_dagger = self._apply_columns(
            matrix.conj().T, adjoint_action
        )
        return acted_dagger.conj().T

    def _apply_k(self, operator, vector):
        return operator.rmatvec(operator.matvec(vector))

    def apply_matrix(self, density):
        density = self._as_matrix(density)
        h_left = self._left(density, self.hamiltonian.matvec)
        h_right = self._right(density, self.hamiltonian.rmatvec)
        result = -1j * (h_left - h_right)

        for channel, operator in self.collapse_operators:
            c_rho = self._left(density, operator.matvec)
            jump = self._right(c_rho, operator.matvec)
            k_left = self._left(
                density,
                lambda value, op=operator: self._apply_k(op, value),
            )
            k_right = self._right(
                density,
                lambda value, op=operator: self._apply_k(op, value),
            )
            result += channel.rate * (
                jump - 0.5 * k_left - 0.5 * k_right
            )
        return result

    def apply_adjoint_matrix(self, observable):
        observable = self._as_matrix(observable)
        h_left = self._left(observable, self.hamiltonian.rmatvec)
        h_right = self._right(observable, self.hamiltonian.matvec)
        result = 1j * (h_left - h_right)

        for channel, operator in self.collapse_operators:
            c_dagger_o = self._left(observable, operator.rmatvec)
            jump = self._right(c_dagger_o, operator.rmatvec)
            k_left = self._left(
                observable,
                lambda value, op=operator: self._apply_k(op, value),
            )
            k_right = self._right(
                observable,
                lambda value, op=operator: self._apply_k(op, value),
            )
            result += channel.rate * (
                jump - 0.5 * k_left - 0.5 * k_right
            )
        return result

    def matvec(self, density_vector):
        return self.apply_matrix(density_vector).reshape(-1, order="F")

    def rmatvec(self, observable_vector):
        return self.apply_adjoint_matrix(observable_vector).reshape(
            -1, order="F"
        )

    @staticmethod
    def frequency_shift(omega, eta):
        """Frequency and damping -> direct-resolvent shift."""
        return complex(float(eta), -float(omega))
