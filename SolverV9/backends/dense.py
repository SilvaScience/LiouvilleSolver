"""Dense Liouville backend for SolverV8."""

from collections import OrderedDict
import threading

import numpy as np

from ..k_integration import integrate_k_response
from ..parallel import (
    effective_n_jobs,
    make_index_blocks,
    parallel_context,
    run_blocks,
)


class DenseLiouvilleBackend:
    """Dense batched Liouville backend."""

    name = "dense"

    def __init__(
        self,
        eta=0.05,
        cache_resolvents=True,
        max_resolvent_cache=None,
        parallel_block_size=None,
    ):
        self.eta = eta
        self.cache_resolvents = cache_resolvents
        self.max_resolvent_cache = max_resolvent_cache
        self.parallel_block_size = parallel_block_size
        self.H_eigen = None
        self.J_plus = None
        self.J_minus = None
        self.c_ops = []
        self.N_k = None
        self.dim = None
        self._initial_state = None
        self._I_super_dense = None
        self._rho_eq_dense = None
        self._trace_vec_dense = None
        self._JL_plus_dense = None
        self._JR_plus_dense = None
        self._JL_minus_dense = None
        self._JR_minus_dense = None
        self._JL_out_dense = None
        self._L_eff_dense = None
        self._dense_resolvent_cache = OrderedDict()
        self._dense_time_cache = OrderedDict()
        self._cache_lock = threading.RLock()

    def build(self, H_eigen, J_plus, J_minus, c_ops, initial_state):
        """Load model operators and rebuild dense Liouville objects."""
        self.H_eigen = H_eigen
        self.J_plus = J_plus
        self.J_minus = J_minus
        self.c_ops = list(c_ops)
        self.N_k, self.dim, _ = H_eigen.shape
        self._initial_state = initial_state
        self.clear_caches()
        self._build_dense_liouville()

    def set_initial_state(self, initial_state):
        """Update rho0 without rebuilding static Liouvillians."""
        self._initial_state = initial_state
        self._rho_eq_dense = initial_state

    def clear_caches(self):
        """Clear dense frequency/time propagator caches."""
        with self._cache_lock:
            self._dense_resolvent_cache.clear()
            self._dense_time_cache.clear()

    def _get_initial_state(self):
        return self._initial_state

    def _spre_dense(self, A):
            """Left-acting dense superoperator: I_d kron A."""
            I = np.eye(self.dim, dtype=np.complex128)
            res = np.einsum("ij,nkl->nikjl", I, A)
            return res.reshape(self.N_k, self.dim**2, self.dim**2)

    def _spost_dense(self, A):
            """Right-acting dense superoperator: A.T kron I_d."""
            I = np.eye(self.dim, dtype=np.complex128)
            res = np.einsum("nji,kl->nikjl", A, I)
            return res.reshape(self.N_k, self.dim**2, self.dim**2)

    def _get_lindblad_dense(self, C, gamma):
            """Build one dense Lindblad dissipator for every k-point."""
            C_dag = np.conj(C.transpose(0, 2, 1))
            C_dag_C = C_dag @ C
            return gamma * (
                self._spre_dense(C) @ self._spost_dense(C_dag)
                - 0.5 * self._spre_dense(C_dag_C)
                - 0.5 * self._spost_dense(C_dag_C)
            )

    def _build_dense_liouville(self):
            """Precompute dense batched superoperators and the static Liouvillian."""
            d2 = self.dim**2
            self._I_super_dense = np.eye(d2, dtype=np.complex128)
            self._rho_eq_dense = self._get_initial_state()

            self._JL_plus_dense = self._spre_dense(self.J_plus)
            self._JR_plus_dense = self._spost_dense(self.J_plus)
            self._JL_minus_dense = self._spre_dense(self.J_minus)
            self._JR_minus_dense = self._spost_dense(self.J_minus)
            self._JL_out_dense = self._spre_dense(self.J_plus + self.J_minus)

            self._trace_vec_dense = np.zeros((self.N_k, 1, d2), dtype=np.complex128)
            for i in range(self.dim):
                self._trace_vec_dense[:, 0, i * self.dim + i] = 1.0

            self._L_eff_dense = (
                self._spre_dense(self.H_eigen) - self._spost_dense(self.H_eigen)
            ).astype(np.complex128)

            for C_eigen, gamma in self.c_ops:
                self._L_eff_dense += 1j * self._get_lindblad_dense(C_eigen, gamma)

    def _get_dense_resolvent(self, w):
            """
            Return cached dense resolvents for all k-points.

            The Liouvillian does not depend on w in this implementation, so each
            frequency only needs one batched inverse per scan.
            """
            key = float(np.round(w, 12))
            with self._cache_lock:
                if self.cache_resolvents:
                    cached = self._dense_resolvent_cache.get(key)
                    if cached is not None:
                        return cached


                A = (w + 1j * self.eta) * self._I_super_dense - self._L_eff_dense
                G = np.linalg.inv(A)


                if not self.cache_resolvents:
                    return G
                self._dense_resolvent_cache[key] = G
                self._dense_resolvent_cache.move_to_end(key)
                if (
                    self.max_resolvent_cache is not None
                    and len(self._dense_resolvent_cache) > self.max_resolvent_cache
                ):
                    self._dense_resolvent_cache.popitem(last=False)
                return G

    def _get_dense_time_propagator(self, delay):
            """Return cached exp(-i L delay) for all k-points."""
            key = float(np.round(delay, 12))
            with self._cache_lock:
                cached = self._dense_time_cache.get(key)
                if cached is not None:
                    return cached


                evals, evecs = np.linalg.eig(-1j * self._L_eff_dense * delay)
                propagator = (
                    evecs * np.exp(evals)[:, np.newaxis, :]
                ) @ np.linalg.inv(evecs)


                self._dense_time_cache[key] = propagator
                return propagator

    def _interaction_dense(self, instruction):
            """Return the dense superoperator for one UFSS instruction."""
            return {
                "Ku": self._JL_plus_dense,
                "Kd": self._JL_minus_dense,
                "Bu": self._JR_minus_dense,
                "Bd": self._JR_plus_dense,
            }[instruction]

    @staticmethod
    def _new_prefix_tree_node():
            return {"children": {}, "pathways": []}

    def _build_pathway_prefix_tree(self, pathways):
            """Build a trie keyed by ordered UFSS interaction labels."""
            root = self._new_prefix_tree_node()
            for pathway in pathways:
                node = root
                for instruction in pathway.interactions:
                    node = node["children"].setdefault(
                        instruction, self._new_prefix_tree_node()
                    )
                node["pathways"].append(pathway)
            return root

    def _apply_interval_dense(self, response, interval, coordinates):
            """Apply the propagation associated with one protocol interval."""
            if interval.domain == "frequency":
                return (
                    self._get_dense_resolvent(float(coordinates[interval.name]))
                    @ response
                )
            if interval.domain == "time":
                return (
                    self._get_dense_time_propagator(
                        float(coordinates[interval.name])
                    )
                    @ response
                )
            return response

    def _trace_output_dense(self, response):
            return (
                self._trace_vec_dense @ (self._JL_out_dense @ response)
            ).reshape(-1)

    def _collect_second_frequency_states(
            self,
            node,
            depth,
            response,
            intervals,
            second_frequency_index,
            coordinates,
            states,
        ):
            """
            Advance the prefix tree until the second frequency propagator.

            The interaction immediately before the second frequency is included,
            because it is independent of the second-axis coordinate. The returned
            states are therefore ready for G(omega_2).
            """
            if depth == second_frequency_index:
                for instruction, child in node["children"].items():
                    next_response = self._interaction_dense(instruction) @ response
                    states.append((child, next_response))
                return

            interval = intervals[depth]
            for instruction, child in node["children"].items():
                next_response = self._interaction_dense(instruction) @ response
                next_response = self._apply_interval_dense(
                    next_response, interval, coordinates
                )
                self._collect_second_frequency_states(
                    child,
                    depth + 1,
                    next_response,
                    intervals,
                    second_frequency_index,
                    coordinates,
                    states,
                )

    def _complete_prefix_tree_spectrum(
            self,
            node,
            depth,
            response,
            intervals,
            coordinates,
            spectra,
            first_index,
            second_index,
            integration_weights,
        ):
            """Finish every suffix below a prefix-tree node."""
            if depth == len(intervals):
                traces = self._trace_output_dense(response)
                for pathway in node["pathways"]:
                    spectra[pathway.name][first_index, second_index] = (
                        integrate_k_response(
                            pathway.response_prefactor * traces,
                            integration_weights,
                            self.N_k,
                            axis=-1,
                        )
                    )
                return

            interval = intervals[depth]
            for instruction, child in node["children"].items():
                next_response = self._interaction_dense(instruction) @ response
                next_response = self._apply_interval_dense(
                    next_response, interval, coordinates
                )
                self._complete_prefix_tree_spectrum(
                    child,
                    depth + 1,
                    next_response,
                    intervals,
                    coordinates,
                    spectra,
                    first_index,
                    second_index,
                    integration_weights,
                )

    def _calculate_first_axis_block(
            self,
            block,
            tree,
            intervals,
            axis_names,
            axis_values,
            fixed_coordinates,
            pathways,
            second_frequency_index,
            integration_weights,
        ):
            """Calculate pathway spectra for a block of first-axis samples."""
            block_results = []
            second_axis_values = axis_values[1]

            for first_index in block:
                first_value = axis_values[0][first_index]
                row_spectra = {
                    pathway.name: np.zeros(
                        (1, second_axis_values.size), dtype=np.complex128
                    )
                    for pathway in pathways
                }
                first_coordinates = {
                    **fixed_coordinates,
                    axis_names[0]: float(first_value),
                }
                pending_states = []
                self._collect_second_frequency_states(
                    tree,
                    0,
                    self._rho_eq_dense,
                    intervals,
                    second_frequency_index,
                    first_coordinates,
                    pending_states,
                )

                for second_index, second_value in enumerate(second_axis_values):
                    coordinates = {
                        **first_coordinates,
                        axis_names[1]: float(second_value),
                    }
                    second_resolvent = self._get_dense_resolvent(second_value)
                    for node, response in pending_states:
                        response_after_second_frequency = second_resolvent @ response
                        self._complete_prefix_tree_spectrum(
                            node,
                            second_frequency_index + 1,
                            response_after_second_frequency,
                            intervals,
                            coordinates,
                            row_spectra,
                            0,
                            second_index,
                            integration_weights,
                        )

                block_results.append(
                    (
                        first_index,
                        {name: values[0].copy() for name, values in row_spectra.items()},
                    )
                )

            return block_results

    def generate_spectrum_prefix_tree(
            self,
            protocol,
            axis_values,
            fixed_coordinates,
            pathways,
            integration_weights,
            parallel_backend="serial",
            n_jobs=1,
            block_size=None,
            blas_threads=None,
        ):
            """Generate arbitrary two-frequency pathway spectra with prefix reuse."""
            axis_names = protocol.frequency_axis_names
            if len(axis_names) != 2:
                raise ValueError("Prefix-tree spectra require two frequency axes.")

            intervals = protocol.intervals
            frequency_indices = [
                index
                for index, interval in enumerate(intervals)
                if interval.domain == "frequency"
            ]
            if len(frequency_indices) != 2:
                raise ValueError("Prefix-tree spectra require two frequency intervals.")
            first_frequency_index, second_frequency_index = frequency_indices
            if second_frequency_index <= first_frequency_index:
                raise ValueError(
                    "Frequency intervals must follow the protocol axis order."
                )

            for pathway in pathways:
                protocol.validate_pathway(pathway)

            tree = self._build_pathway_prefix_tree(pathways)
            axis_values = tuple(np.asarray(values, dtype=float) for values in axis_values)
            shape = (len(axis_values[0]), len(axis_values[1]))
            spectra = {
                pathway.name: np.zeros(shape, dtype=np.complex128)
                for pathway in pathways
            }
            fixed_coordinates = dict(fixed_coordinates)
            n_jobs_eff = effective_n_jobs(n_jobs)
            blocks = make_index_blocks(
                shape[0],
                n_jobs_eff,
                block_size,
                self.parallel_block_size,
            )

            def worker(block):
                return self._calculate_first_axis_block(
                    block,
                    tree,
                    intervals,
                    axis_names,
                    axis_values,
                    fixed_coordinates,
                    pathways,
                    second_frequency_index,
                    integration_weights,
                )

            with parallel_context(blas_threads):
                block_results = run_blocks(
                    blocks,
                    worker,
                    parallel_backend=parallel_backend,
                    n_jobs=n_jobs_eff,
                )

            for first_index, row_spectra in block_results:
                for name, values in row_spectra.items():
                    spectra[name][first_index, :] = values

            return spectra

    def calc_pathway(self, pathway, protocol, coordinates):
            """Evaluate one arbitrary-order pathway with the dense backend."""
            protocol.validate_pathway(pathway)
            coordinates = dict(coordinates)
            required = {
                interval.name
                for interval in protocol.intervals
                if interval.domain != "identity"
            }
            missing = sorted(required.difference(coordinates))
            if missing:
                raise KeyError(f"Missing protocol coordinate(s): {missing}")

            response = self._rho_eq_dense
            for instruction, interval in zip(
                pathway.interactions, protocol.intervals
            ):
                response = self._interaction_dense(instruction) @ response
                response = self._apply_interval_dense(
                    response, interval, coordinates
                )
            traces = self._trace_output_dense(response)
            return pathway.response_prefactor * traces
