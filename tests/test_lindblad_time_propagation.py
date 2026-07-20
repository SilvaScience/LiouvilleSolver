import unittest

import numpy as np

from SolverV9.backends.dense import DenseLiouvilleBackend


def _vectorize(matrix):
    return np.asarray(matrix, dtype=np.complex128).T.reshape(-1, 1)


def _unvectorize(vector, dimension):
    return np.asarray(vector).reshape(dimension, dimension).T


def _build_backend(
    dimension,
    collapse_operators,
    initial_density_matrix,
    hamiltonian=None,
):
    zeros = np.zeros((1, dimension, dimension), dtype=np.complex128)
    if hamiltonian is None:
        hamiltonian = zeros
    else:
        hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)[
            np.newaxis, :, :
        ]
    backend = DenseLiouvilleBackend(eta=0.0)
    backend.build(
        H_eigen=hamiltonian,
        J_plus=zeros,
        J_minus=zeros,
        c_ops=[
            (operator[np.newaxis, :, :], rate)
            for operator, rate in collapse_operators
        ],
        initial_state=_vectorize(initial_density_matrix)[np.newaxis, :, :],
    )
    return backend


def _propagate_density_matrix(backend, density_matrix, delay):
    dimension = density_matrix.shape[0]
    propagated = backend._get_dense_time_propagator(delay)[0] @ _vectorize(
        density_matrix
    )
    return _unvectorize(propagated, dimension)


class LindbladTimePropagationTests(unittest.TestCase):
    def test_coherent_evolution_matches_analytic_phase(self):
        energy = 0.7
        delay = 2.3
        hamiltonian = np.diag([0.0, energy])
        coherent_state = 0.5 * np.array(
            [[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128
        )
        backend = _build_backend(
            2,
            [],
            coherent_state,
            hamiltonian=hamiltonian,
        )

        result = _propagate_density_matrix(backend, coherent_state, delay)
        expected = coherent_state.copy()
        expected[0, 1] *= np.exp(1j * energy * delay)
        expected[1, 0] *= np.exp(-1j * energy * delay)

        np.testing.assert_allclose(result, expected, atol=1e-13)

    def test_two_level_amplitude_damping_matches_analytic_solution(self):
        gamma = 0.3
        delay = 5.0
        lowering = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
        excited_state = np.diag([0.0, 1.0]).astype(np.complex128)
        backend = _build_backend(2, [(lowering, gamma)], excited_state)

        result = _propagate_density_matrix(backend, excited_state, delay)
        expected_excited_population = np.exp(-gamma * delay)

        np.testing.assert_allclose(
            np.diag(result),
            [1.0 - expected_excited_population, expected_excited_population],
            atol=1e-13,
        )
        np.testing.assert_allclose(np.trace(result), 1.0, atol=1e-13)

    def test_equal_rate_three_level_cascade_handles_jordan_block(self):
        gamma = 1.0
        delay = 1.0
        jump_1_to_0 = np.zeros((3, 3), dtype=np.complex128)
        jump_2_to_1 = np.zeros((3, 3), dtype=np.complex128)
        jump_1_to_0[0, 1] = 1.0
        jump_2_to_1[1, 2] = 1.0
        upper_state = np.diag([0.0, 0.0, 1.0]).astype(np.complex128)
        backend = _build_backend(
            3,
            [(jump_1_to_0, gamma), (jump_2_to_1, gamma)],
            upper_state,
        )

        result = _propagate_density_matrix(backend, upper_state, delay)
        exponential = np.exp(-gamma * delay)
        expected_populations = [
            1.0 - (1.0 + gamma * delay) * exponential,
            gamma * delay * exponential,
            exponential,
        ]

        np.testing.assert_allclose(np.diag(result), expected_populations, atol=1e-13)
        np.testing.assert_allclose(np.trace(result), 1.0, atol=1e-13)
        self.assertGreaterEqual(np.min(np.linalg.eigvalsh(result)), -1e-13)

    def test_time_propagator_is_cached_for_equivalent_delay_values(self):
        lowering = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
        ground_state = np.diag([1.0, 0.0]).astype(np.complex128)
        backend = _build_backend(2, [(lowering, 0.2)], ground_state)

        first = backend._get_dense_time_propagator(1.0)
        second = backend._get_dense_time_propagator(1.0 + 1e-14)

        self.assertIs(second, first)
        self.assertEqual(len(backend._dense_time_cache), 1)


if __name__ == "__main__":
    unittest.main()
