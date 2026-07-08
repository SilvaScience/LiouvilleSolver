"""Dense Liouville backend for SolverV8."""

from collections import OrderedDict

from joblib import Parallel, delayed
import numpy as np

from ..k_integration import integrate_k_response
from ..pathways import default_frequency_pathways
from ..parallel import make_w3_blocks


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
        self._dense_resolvent_cache.clear()
        self._dense_time_cache.clear()

    def _get_initial_state(self):
        return self._initial_state

    @staticmethod
    def _wants_rephasing(spectrum_components):
        return spectrum_components in {"both", "rephasing"}

    @staticmethod
    def _wants_unrephasing(spectrum_components):
        return spectrum_components in {"both", "unrephasing"}

    @staticmethod
    def _format_spectra_result(S3_reph, S3_unreph, spectrum_components):
        result = {}
        if DenseLiouvilleBackend._wants_rephasing(spectrum_components):
            result["rephasing"] = S3_reph
        if DenseLiouvilleBackend._wants_unrephasing(spectrum_components):
            result["unrephasing"] = S3_unreph
        if spectrum_components == "both":
            result["absorptive"] = S3_reph + S3_unreph
        return result

    def supports_default_2d_fast_path(self, pathways):
        return list(pathways) == default_frequency_pathways()

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
            cached = self._dense_time_cache.get(key)
            if cached is not None:
                return cached


            evals, evecs = np.linalg.eig(-1j * self._L_eff_dense * delay)
            propagator = (
                evecs * np.exp(evals)[:, np.newaxis, :]
            ) @ np.linalg.inv(evecs)


            self._dense_time_cache[key] = propagator
            return propagator

    def _precompute_rephasing_dense_rhs(self, w_list, G_list, G2):
            """Precompute dense rephasing RHS vectors that only depend on w1."""
            n_w = len(w_list)
            d2 = self.dim**2
            rhs = np.empty((3, n_w, self.N_k, d2, 1), dtype=np.complex128)
            source = self._JR_minus_dense @ self._rho_eq_dense

            for j, G1 in enumerate(G_list):
                v1 = G1 @ source

                mid_gsb = G2 @ (self._JR_plus_dense @ v1)
                rhs[0, j] = self._JL_plus_dense @ mid_gsb

                mid_se_esa = G2 @ (self._JL_plus_dense @ v1)
                rhs[1, j] = self._JR_plus_dense @ mid_se_esa
                rhs[2, j] = self._JL_plus_dense @ mid_se_esa

            return rhs

    def _precompute_unrephasing_dense_rhs(self, w_list, G_list, G2):
            """Precompute dense non-rephasing RHS vectors that only depend on w1."""
            n_w = len(w_list)
            d2 = self.dim**2
            rhs = np.empty((3, n_w, self.N_k, d2, 1), dtype=np.complex128)
            source = self._JL_plus_dense @ self._rho_eq_dense

            for j, G1 in enumerate(G_list):
                v1 = G1 @ source

                mid_gsb = G2 @ (self._JL_minus_dense @ v1)
                rhs[0, j] = self._JL_plus_dense @ mid_gsb

                mid_se_esa = G2 @ (self._JR_minus_dense @ v1)
                rhs[1, j] = self._JR_plus_dense @ mid_se_esa
                rhs[2, j] = self._JL_plus_dense @ mid_se_esa

            return rhs

    def _scan_dense_w3_block(self, block, G_list, rhs, integration_weights):
            """Apply one block of precomputed w3 resolvents to immutable RHS data."""
            n_w = len(G_list)
            trace_vec = self._trace_vec_dense[np.newaxis, np.newaxis, ...]
            output_op = self._JL_out_dense[np.newaxis, np.newaxis, ...]
            columns = []

            for i in block:
                G3 = G_list[i]
                paths = G3[np.newaxis, np.newaxis, ...] @ rhs
                traces = (trace_vec @ (output_op @ paths)).reshape(3, n_w, self.N_k)
                column = -1j * integrate_k_response(
                    traces[0] + traces[1] - traces[2],
                    integration_weights,
                    self.N_k,
                    axis=1,
                )
                columns.append((i, column))

            return columns

    def _scan_dense_component_from_rhs(
            self,
            G_list,
            rhs,
            integration_weights,
            parallel_backend="serial",
            n_jobs=1,
            block_size=None,
        ):
            """Apply every w3 resolvent to precomputed RHS vectors."""
            n_w = len(G_list)
            spectrum = np.empty((n_w, n_w), dtype=np.complex128)
            blocks = make_w3_blocks(n_w, n_jobs, block_size, self.parallel_block_size)

            if parallel_backend == "threading" and n_jobs > 1:
                nested = Parallel(n_jobs=n_jobs, backend="threading")(
                    delayed(self._scan_dense_w3_block)(
                        block, G_list, rhs, integration_weights
                    )
                    for block in blocks
                )
                columns = [item for block_result in nested for item in block_result]
            else:
                columns = [
                    item
                    for block in blocks
                    for item in self._scan_dense_w3_block(
                        block, G_list, rhs, integration_weights
                    )
                ]

            for i, column in columns:
                spectrum[:, i] = column

            return spectrum

    def generate_default_2d(
            self,
            w_list,
            tau2,
            integration_weights,
            spectrum_components,
            parallel_backend="serial",
            n_jobs=1,
            block_size=None,
        ):
            """Optimized dense 2D scan that reuses all w1-dependent RHS vectors."""
            G_list = [self._get_dense_resolvent(w) for w in w_list]
            G2 = self._get_dense_time_propagator(tau2)

            S3_reph = None
            S3_unreph = None

            if self._wants_rephasing(spectrum_components):
                rhs_reph = self._precompute_rephasing_dense_rhs(w_list, G_list, G2)
                S3_reph = self._scan_dense_component_from_rhs(
                    G_list,
                    rhs_reph,
                    integration_weights,
                    parallel_backend,
                    n_jobs,
                    block_size,
                )

            if self._wants_unrephasing(spectrum_components):
                rhs_unreph = self._precompute_unrephasing_dense_rhs(w_list, G_list, G2)
                S3_unreph = self._scan_dense_component_from_rhs(
                    G_list,
                    rhs_unreph,
                    integration_weights,
                    parallel_backend,
                    n_jobs,
                    block_size,
                )

            return self._format_spectra_result(
                S3_reph, S3_unreph, spectrum_components
            )

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

    def generate_spectrum_prefix_tree(
            self,
            protocol,
            axis_values,
            fixed_coordinates,
            pathways,
            integration_weights,
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

            for first_index, first_value in enumerate(axis_values[0]):
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

                for second_index, second_value in enumerate(axis_values[1]):
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
                            spectra,
                            first_index,
                            second_index,
                            integration_weights,
                        )

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
