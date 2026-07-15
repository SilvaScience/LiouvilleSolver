"""Public SolverV8 orchestration layer."""

import numpy as np

from .backends.dense import DenseLiouvilleBackend
from .k_integration import integrate_k_response, resolve_k_weights
from .models import FrequencyPathway, SpectrumResult, SpectroscopyProtocol
from .pathways import (
    generate_pathways_with_ufss as build_pathways_with_ufss,
    make_pathway_names_unique,
    phase_discrimination_component,
    translate_ufss_diagrams as translate_ufss_diagrams_function,
)


class LiouvilleSpectroscopySolver:
    """Universal Liouville-space solver for impulsive spectroscopy."""

    _BACKEND_CLASSES = {"dense": DenseLiouvilleBackend}

    def __init__(self, params):
        self.params = params
        self.eta = params.get("Eta", 0.05)
        self.T = params.get("T", 0.01)
        self.rwa_tol = params.get("rwa_tol", 1e-6)
        self.cache_resolvents = params.get("cache_resolvents", True)
        self.max_resolvent_cache = params.get("max_resolvent_cache", None)
        self.backend = str(params.get("backend", "dense")).lower()
        self.parallel_backend = params.get("parallel_backend", "serial")
        self.parallel_block_size = params.get("parallel_block_size", None)
        self.blas_threads = params.get("blas_threads", None)
        self.n_jobs = params.get("n_jobs", 1)
        self.density_matrix_tolerance = params.get(
            "density_matrix_tolerance", 1e-10
        )
        self._active_backend = None
        self._backend = None

        self.H_eigen = None
        self.energies = None
        self.eigenvectors = None
        self.J_plus = None
        self.J_minus = None

        self.c_ops = []
        self.c_ops_eigen = []

        self.dim = None
        self.N_k = None

        self._initial_density_matrix_eigen = None
        self._pending_initial_density_matrix = params.get(
            "initial_density_matrix", params.get("rho0", None)
        )
        self._pending_density_matrix_basis = params.get(
            "density_matrix_basis", "site"
        )
        self.pathways = []

    def _phase_discrimination_component(self, phase_discrimination):
        return phase_discrimination_component(phase_discrimination)

    def _make_pathway_names_unique(self, pathways):
        return make_pathway_names_unique(pathways)

    def translate_ufss_diagrams(self, *args, **kwargs):
        return translate_ufss_diagrams_function(*args, **kwargs)

    def generate_pathways_with_ufss(self, *args, replace=False, **kwargs):
        pathways = build_pathways_with_ufss(*args, **kwargs)
        if replace:
            self.set_pathways(pathways)
        return pathways

    def set_pathways_from_ufss(self, diagrams, **kwargs):
        pathways = self.translate_ufss_diagrams(diagrams, **kwargs)
        self.set_pathways(pathways)
        return pathways

    def get_pathways(self, component=None):
            """Return configured pathways, optionally filtered by component."""
            if component is None:
                return list(self.pathways)
            component = str(component).lower().replace("-", "")
            if component in {"nonrephasing", "nonrephase"}:
                component = "unrephasing"
            return [pathway for pathway in self.pathways if pathway.component == component]

    def pathway_summary(self, component=None):
            """Return the active pathway definitions in a notebook-friendly form."""
            return [
                {
                    "name": pathway.name,
                    "component": pathway.component,
                    "interactions": pathway.interactions,
                    "pulse_indices": pathway.pulse_indices,
                    "amplitude": pathway.amplitude,
                    "prefactor": pathway.prefactor,
                    "response_prefactor": pathway.response_prefactor,
                    "coherence_orders": pathway.coherence_orders,
                    "bra_sign": pathway.bra_sign,
                    "coefficient": pathway.coefficient,
                }
                for pathway in self.get_pathways(component)
            ]

    def set_pathways(self, pathways):
            """
            Replace the active pathway list.

            Entries may be :class:`FrequencyPathway` objects or dictionaries with
            matching constructor keys.
            """
            normalized = []
            names = set()
            for item in pathways:
                pathway = item if isinstance(item, FrequencyPathway) else FrequencyPathway(**item)
                if pathway.name in names:
                    raise ValueError(f"Duplicate pathway name: {pathway.name!r}")
                names.add(pathway.name)
                normalized.append(pathway)
            if not normalized:
                raise ValueError("At least one pathway is required.")
            self.pathways = normalized

    def _clear_backend_caches(self):
            if self._backend is not None:
                self._backend.clear_caches()

    def _clean_gamma(self, gamma):
            """Return a real scalar Lindblad rate."""
            if gamma is None:
                return None

            gamma = np.asarray(gamma).item()
            gamma = np.real_if_close(gamma)
            if np.iscomplexobj(gamma):
                if abs(np.imag(gamma)) > 1e-12:
                    raise ValueError(f"Lindblad rates must be real, got {gamma}")
                gamma = np.real(gamma)
            return float(gamma)

    def _as_k_stack(self, array, name):
            """
            Normalize a matrix-like object to shape (N_k, d, d).

            Accepted input shapes are:
            - (d, d)
            - (N_k, d, d)
            """
            array = np.asarray(array, dtype=np.complex128)

            if array.ndim == 2:
                array = array[np.newaxis, :, :]
            elif array.ndim != 3:
                raise ValueError(
                    f"{name} must have shape (d, d) or (N_k, d, d); got {array.shape}"
                )

            if array.shape[1] != array.shape[2]:
                raise ValueError(f"{name} must contain square matrices; got {array.shape}")

            return array

    def _broadcast_stack(self, array, name, N_k, dim):
            """Broadcast a single-k matrix stack and validate all dimensions."""
            if array.shape[1:] != (dim, dim):
                raise ValueError(
                    f"{name} has matrix shape {array.shape[1:]}, expected {(dim, dim)}"
                )

            if array.shape[0] == N_k:
                return array

            if array.shape[0] == 1 and N_k > 1:
                return np.repeat(array, N_k, axis=0)

            raise ValueError(f"{name} has N_k={array.shape[0]}, expected {N_k}")

    def _split_c_ops(self, c_ops_raw):
            """
            Normalize collapse operators without deciding the final N_k yet.

            Each item can be either:
            - C
            - (C, gamma)
            """
            if c_ops_raw is None:
                return []

            c_ops_prepared = []
            for idx, item in enumerate(c_ops_raw):
                if isinstance(item, tuple) and len(item) == 2:
                    C_raw, gamma = item
                else:
                    C_raw, gamma = item, None

                C_raw = self._as_k_stack(C_raw, f"c_ops_raw[{idx}]")
                c_ops_prepared.append((C_raw, self._clean_gamma(gamma)))

            return c_ops_prepared

    def _prepare_model_inputs(self, H_model, interaction_op_array, c_ops_raw):
            """Validate and broadcast all model inputs to a common shape."""
            H_model = self._as_k_stack(H_model, "H_model")
            interaction_op_array = self._as_k_stack(
                interaction_op_array, "interaction_op_array"
            )
            c_ops_prepared = self._split_c_ops(c_ops_raw)

            dim = H_model.shape[1]
            candidate_N_k = [H_model.shape[0], interaction_op_array.shape[0]]
            candidate_N_k.extend(C_raw.shape[0] for C_raw, _ in c_ops_prepared)
            N_k = max(candidate_N_k)

            H_model = self._broadcast_stack(H_model, "H_model", N_k, dim)
            interaction_op_array = self._broadcast_stack(
                interaction_op_array, "interaction_op_array", N_k, dim
            )

            c_ops_broadcasted = []
            for idx, (C_raw, gamma) in enumerate(c_ops_prepared):
                C_raw = self._broadcast_stack(C_raw, f"c_ops_raw[{idx}]", N_k, dim)
                c_ops_broadcasted.append((C_raw, gamma))

            return H_model, interaction_op_array, c_ops_broadcasted

    def _matrix_stack_to_vectors(self, matrices):
            """Column-vectorize a matrix stack consistently with ``I kron A``."""
            return matrices.transpose(0, 2, 1).reshape(
                self.N_k, self.dim**2, 1
            )

    def _vectors_to_matrix_stack(self, vectors):
            """Inverse of :meth:`_matrix_stack_to_vectors`."""
            return np.asarray(vectors).reshape(
                self.N_k, self.dim, self.dim
            ).transpose(0, 2, 1)

    def set_initial_density_matrix(
            self,
            rho,
            basis="site",
            normalize=True,
            validate=True,
        ):
            """
            Set the initial density matrix used by every response pathway.

            Parameters
            ----------
            rho : array_like
                Shape ``(d, d)`` or ``(N_k, d, d)``. A single matrix is
                broadcast to every k-point.
            basis : {"site", "eigen"}
                Basis of the supplied matrix. Site-basis matrices are transformed
                with the eigenvectors stored by :meth:`feed_model`.
            normalize : bool
                Normalize each k-point to unit trace.
            validate : bool
                Check Hermiticity, unit trace, and positive semidefiniteness.
            """
            if self.N_k is None or self.dim is None:
                raise RuntimeError(
                    "Call feed_model() before set_initial_density_matrix(), or "
                    "pass initial_density_matrix directly to feed_model()."
                )
            if basis not in {"site", "eigen"}:
                raise ValueError("basis must be either 'site' or 'eigen'")

            rho_stack = self._as_k_stack(rho, "rho")
            rho_stack = self._broadcast_stack(
                rho_stack, "rho", self.N_k, self.dim
            ).copy()

            if basis == "site":
                rho_eigen = np.empty_like(rho_stack)
                for i_k in range(self.N_k):
                    U = self.eigenvectors[i_k]
                    rho_eigen[i_k] = U.conj().T @ rho_stack[i_k] @ U
            else:
                rho_eigen = rho_stack

            tol = float(self.density_matrix_tolerance)
            for i_k, rho_k in enumerate(rho_eigen):
                if validate and not np.allclose(
                    rho_k, rho_k.conj().T, atol=tol, rtol=0
                ):
                    raise ValueError(
                        f"rho[{i_k}] is not Hermitian within tolerance {tol}."
                    )

                trace = np.trace(rho_k)
                if abs(trace.imag) > tol or trace.real <= tol:
                    raise ValueError(
                        f"rho[{i_k}] must have a positive real trace; got {trace}."
                    )
                if normalize:
                    rho_eigen[i_k] = rho_k / trace.real
                elif validate and not np.isclose(
                    trace.real, 1.0, atol=tol, rtol=0
                ):
                    raise ValueError(
                        f"rho[{i_k}] has trace {trace.real}; expected 1."
                    )

                if validate:
                    eigenvalues = np.linalg.eigvalsh(
                        0.5 * (rho_eigen[i_k] + rho_eigen[i_k].conj().T)
                    )
                    if np.min(eigenvalues) < -tol:
                        raise ValueError(
                            f"rho[{i_k}] is not positive semidefinite; minimum "
                            f"eigenvalue is {np.min(eigenvalues)}."
                        )

            self._initial_density_matrix_eigen = rho_eigen
            self._refresh_initial_state_vectors()

    def clear_initial_density_matrix(self):
            """Return to the default thermal (or T=0 ground-state) density matrix."""
            self._initial_density_matrix_eigen = None
            self._pending_initial_density_matrix = None
            self._refresh_initial_state_vectors()

    def get_initial_density_matrix(self, basis="eigen"):
            """Return the active normalized density matrix as a matrix stack."""
            if self.N_k is None:
                raise RuntimeError("Call feed_model() before requesting rho0.")
            if basis not in {"site", "eigen"}:
                raise ValueError("basis must be either 'site' or 'eigen'")

            matrices = self._vectors_to_matrix_stack(self._get_initial_state())
            if basis == "eigen":
                return matrices.copy()

            rho_site = np.empty_like(matrices)
            for i_k in range(self.N_k):
                U = self.eigenvectors[i_k]
                rho_site[i_k] = U @ matrices[i_k] @ U.conj().T
            return rho_site

    def _get_initial_state(self):
            """Return the configured or default initial state in vectorized form."""
            if self._initial_density_matrix_eigen is None:
                return self._get_thermal_state()
            return self._matrix_stack_to_vectors(
                self._initial_density_matrix_eigen
            )

    def _refresh_initial_state_vectors(self):
            """Update backend-specific rho vectors without rebuilding Liouvillians."""
            if self.N_k is None or self.dim is None:
                return
            if self._backend is not None:
                self._backend.set_initial_state(self._get_initial_state())

    def feed_model(
            self,
            H_model,
            interaction_op_array,
            c_ops_raw=None,
            initial_density_matrix=None,
            density_matrix_basis=None,
        ):
            """
            Load a model in the site basis and prepare dense Liouville operators.

            H_model and interaction_op_array must have shape (d, d) or (N_k, d, d).
            The interaction operator may represent, for example, a dipole or a
            current. Its raising and lowering parts are determined from energy
            differences in the eigenbasis. Diagonal and quasi-degenerate matrix
            elements within rwa_tol are excluded.
            c_ops_raw may contain raw matrices or (matrix, gamma) tuples. If gammas
            are provided here, the dissipation is configured immediately.
            initial_density_matrix may be supplied in the site or eigen basis. If
            omitted, the thermal state determined by T and mu is used.
            """
            H_model, interaction_op_array, c_ops_prepared = self._prepare_model_inputs(
                H_model, interaction_op_array, c_ops_raw
            )

            print("--- Model loading ---")

            self.N_k, self.dim, _ = H_model.shape
            self.energies = np.zeros((self.N_k, self.dim), dtype=float)
            self.eigenvectors = np.zeros(
                (self.N_k, self.dim, self.dim), dtype=np.complex128
            )
            self.H_eigen = np.zeros(
                (self.N_k, self.dim, self.dim), dtype=np.complex128
            )
            self.J_plus = np.zeros(
                (self.N_k, self.dim, self.dim), dtype=np.complex128
            )
            self.J_minus = np.zeros(
                (self.N_k, self.dim, self.dim), dtype=np.complex128
            )
            self.c_ops_eigen = [
                np.zeros((self.N_k, self.dim, self.dim), dtype=np.complex128)
                for _ in c_ops_prepared
            ]

            for i_k in range(self.N_k):
                evals, evecs = np.linalg.eigh(H_model[i_k])
                evals = np.real_if_close(evals).real

                self.energies[i_k] = evals
                self.eigenvectors[i_k] = evecs
                self.H_eigen[i_k] = np.diag(evals)

                U = evecs
                U_dag = U.conj().T

                O_eigen = U_dag @ interaction_op_array[i_k] @ U

                delta_E = evals[:, np.newaxis] - evals[np.newaxis, :]
                self.J_plus[i_k] = np.where(
                    delta_E > self.rwa_tol, O_eigen, 0.0
                )
                self.J_minus[i_k] = np.where(
                    delta_E < -self.rwa_tol, O_eigen, 0.0
                )

                for idx, (C_raw, _) in enumerate(c_ops_prepared):
                    self.c_ops_eigen[idx][i_k] = U_dag @ C_raw[i_k] @ U

            gammas = [gamma for _, gamma in c_ops_prepared]
            if any(gamma is not None for gamma in gammas):
                if any(gamma is None for gamma in gammas):
                    raise ValueError(
                        "Either provide a gamma for every collapse operator or call "
                        "set_dissipation() after feed_model()."
                    )
                self.c_ops = [
                    (self.c_ops_eigen[idx], gamma) for idx, gamma in enumerate(gammas)
                ]
            else:
                self.c_ops = []

            if initial_density_matrix is None:
                initial_density_matrix = self._pending_initial_density_matrix
            if density_matrix_basis is None:
                density_matrix_basis = self._pending_density_matrix_basis
            self._initial_density_matrix_eigen = None
            if initial_density_matrix is not None:
                self.set_initial_density_matrix(
                    initial_density_matrix,
                    basis=density_matrix_basis,
                )

            self._build_liouville_backend()
            print("Model transformed to the eigenbasis.")
            print(f"Liouville backend ready: {self._active_backend}.")

    def set_dissipation(self, c_ops_list, basis="eigen"):
            """
            Define Lindblad jump operators.

            Parameters
            ----------
            c_ops_list : list
                List of (matrix_stack, gamma) pairs.
            basis : {"eigen", "site"}
                Use "eigen" when the operators are already projected. Use "site"
                to project them with the eigenvectors stored by feed_model().
            """
            if self.N_k is None or self.dim is None:
                raise RuntimeError("Call feed_model() before set_dissipation().")

            if basis not in {"eigen", "site"}:
                raise ValueError("basis must be either 'eigen' or 'site'")

            c_ops_eigen = []
            c_ops_with_gamma = []

            for idx, item in enumerate(c_ops_list):
                if not (isinstance(item, tuple) and len(item) == 2):
                    raise ValueError(
                        "set_dissipation expects a list of (matrix_stack, gamma) pairs."
                    )

                C_raw, gamma = item
                gamma = self._clean_gamma(gamma)
                C_raw = self._as_k_stack(C_raw, f"c_ops_list[{idx}]")
                C_raw = self._broadcast_stack(
                    C_raw, f"c_ops_list[{idx}]", self.N_k, self.dim
                )

                if basis == "site":
                    C_eigen = np.zeros_like(C_raw, dtype=np.complex128)
                    for i_k in range(self.N_k):
                        U = self.eigenvectors[i_k]
                        C_eigen[i_k] = U.conj().T @ C_raw[i_k] @ U
                else:
                    C_eigen = C_raw

                c_ops_eigen.append(C_eigen)
                c_ops_with_gamma.append((C_eigen, gamma))

            self.c_ops_eigen = c_ops_eigen
            self.c_ops = c_ops_with_gamma
            self._build_liouville_backend()
            print("Dissipation operators updated.")

    def _select_backend(self):
            """Validate and return the requested Liouville backend."""
            if self.backend not in self._BACKEND_CLASSES:
                supported = ", ".join(sorted(self._BACKEND_CLASSES))
                raise ValueError(
                    f"Unsupported Liouville backend {self.backend!r}. "
                    f"Available backend(s): {supported}."
                )
            return self.backend

    def _build_liouville_backend(self):
            """Build the selected backend through the backend registry."""
            self._active_backend = self._select_backend()
            backend_cls = self._BACKEND_CLASSES[self._active_backend]
            self._backend = backend_cls(
                eta=self.eta,
                cache_resolvents=self.cache_resolvents,
                max_resolvent_cache=self.max_resolvent_cache,
                parallel_block_size=self.parallel_block_size,
            )
            self._backend.build(
                self.H_eigen,
                self.J_plus,
                self.J_minus,
                self.c_ops,
                self._get_initial_state(),
            )

    def _get_thermal_state(self):
            """Compute the vectorized initial thermal equilibrium state."""
            d = self.dim
            d2 = d**2
            kB = 8.6173e-5
            mu = self.params.get("mu", 0.0)

            if self.T > 0:
                beta = 1.0 / (kB * self.T)
                shifted = self.energies - mu
                shifted = shifted - np.min(shifted, axis=1, keepdims=True)
                weights = np.exp(-beta * shifted)
                rho_diag = weights / np.sum(weights, axis=1, keepdims=True)
            else:
                rho_diag = np.zeros((self.N_k, d), dtype=float)
                rho_diag[:, 0] = 1.0

            rho_vec = np.zeros((self.N_k, d2, 1), dtype=np.complex128)
            for i in range(d):
                rho_vec[:, i * d + i, 0] = rho_diag[:, i]

            return rho_vec

    def _resolve_pathway(self, pathway):
            """Resolve a pathway name, dictionary, or FrequencyPathway object."""
            if isinstance(pathway, FrequencyPathway):
                return pathway
            if isinstance(pathway, dict):
                return FrequencyPathway(**pathway)
            if isinstance(pathway, str):
                matches = [item for item in self.pathways if item.name == pathway]
                if not matches:
                    available = ", ".join(item.name for item in self.pathways)
                    raise KeyError(
                        f"Unknown pathway {pathway!r}. Available pathways: {available}"
                    )
                return matches[0]
            raise TypeError(
                "pathway must be a name, dictionary, or FrequencyPathway."
            )

    def calc_pathway(
            self,
            pathway,
            protocol,
            coordinates,
        ):
            """Evaluate one pathway using an explicit protocol and coordinates."""
            if self._active_backend is None:
                raise RuntimeError("Call feed_model() before calc_pathway().")
            pathway = self._resolve_pathway(pathway)
            if not isinstance(protocol, SpectroscopyProtocol):
                raise TypeError("protocol must be a SpectroscopyProtocol.")
            coordinates = dict(coordinates)
            return self._backend.calc_pathway(pathway, protocol, coordinates)

    def calc_component(
            self,
            component,
            protocol,
            coordinates,
            pathways=None,
        ):
            """Sum all selected pathways belonging to one component."""
            normalized_component = str(component).lower().replace("-", "")
            if normalized_component in {"nonrephasing", "nonrephase"}:
                normalized_component = "unrephasing"
            candidates = (
                self.pathways
                if pathways is None
                else [self._resolve_pathway(item) for item in pathways]
            )
            selected = [
                pathway
                for pathway in candidates
                if pathway.component == normalized_component
            ]
            if not selected:
                raise ValueError(f"No pathways found for component {component!r}.")
            response = np.zeros(self.N_k, dtype=np.complex128)
            for pathway in selected:
                response += self.calc_pathway(pathway, protocol, coordinates)
            return response

    def _resolve_k_weights(self, k_array=None, k_weights=None):
            """Return validated per-k integration weights.

            Explicit ``k_weights`` are used as supplied and must already include
            the desired integration measure, such as ``dk / (2*pi)``.  When they
            are omitted, the historical scalar rule derived from ``k_array`` is
            retained for backward compatibility.  A model without either input
            keeps the historical unweighted sum over k.
            """
            return resolve_k_weights(self.N_k, k_array, k_weights)

    def _integrate_k_response(self, response, k_weights, axis=-1):
            """Integrate one response array along its momentum axis."""
            response = np.asarray(response)
            axis = int(axis)
            if axis < 0:
                axis += response.ndim
            if axis < 0 or axis >= response.ndim:
                raise ValueError(
                    f"axis {axis} is invalid for a {response.ndim}D response"
                )
            if response.shape[axis] != self.N_k:
                raise ValueError(
                    f"response momentum axis has length {response.shape[axis]}, "
                    f"expected {self.N_k}"
                )
            shape = [1] * response.ndim
            shape[axis] = self.N_k
            return integrate_k_response(response, k_weights, self.N_k, axis=axis)

    def generate_spectrum(
            self,
            protocol,
            axes,
            delays,
            pathways=None,
            k_array=None,
            verbose=True,
            k_weights=None,
            n_jobs=None,
            parallel_backend=None,
            block_size=None,
            blas_threads=None,
        ):
            """Generate a generic two-frequency spectrum.

            ``k_weights`` optionally supplies one complete integration weight per
            k-point.  If omitted, ``k_array`` retains the legacy rectangular rule.
            """
            if self.N_k is None:
                raise RuntimeError("Call feed_model() before generate_spectrum().")
            if not isinstance(protocol, SpectroscopyProtocol):
                raise TypeError("protocol must be a SpectroscopyProtocol.")
            axis_names = protocol.frequency_axis_names
            if len(axis_names) != 2:
                raise ValueError(
                    "generate_spectrum requires exactly two frequency intervals."
                )
            axes = dict(axes)
            if set(axes) != set(axis_names):
                raise ValueError(
                    f"axes must contain exactly {axis_names}; got {tuple(axes)}."
                )
            axis_values = tuple(
                np.asarray(axes[name], dtype=float) for name in axis_names
            )
            if any(values.ndim != 1 or values.size == 0 for values in axis_values):
                raise ValueError("Frequency axes must be non-empty 1D arrays.")

            delays = dict(delays)
            required_delays = set(protocol.time_interval_names)
            missing_delays = sorted(required_delays.difference(delays))
            extra_delays = sorted(set(delays).difference(required_delays))
            if missing_delays or extra_delays:
                raise ValueError(
                    f"delays mismatch; missing={missing_delays}, extra={extra_delays}."
                )
            fixed_coordinates = {
                name: float(delays[name]) for name in protocol.time_interval_names
            }
            selected = (
                list(self.pathways)
                if pathways is None
                else [self._resolve_pathway(item) for item in pathways]
            )
            if not selected:
                raise ValueError("At least one pathway must be selected.")
            for pathway in selected:
                protocol.validate_pathway(pathway)

            integration_weights = self._resolve_k_weights(k_array, k_weights)
            shape = (len(axis_values[0]), len(axis_values[1]))
            self._clear_backend_caches()
            if n_jobs is None:
                n_jobs = self.n_jobs
            if parallel_backend is None:
                parallel_backend = self.parallel_backend
            if blas_threads is None:
                blas_threads = self.blas_threads
            if verbose:
                print(
                    f"Calculating {len(selected)} pathway spectrum/s on a "
                    f"{shape[0]}x{shape[1]} grid with protocol {protocol.name!r} "
                    f"using parallel={parallel_backend}, n_jobs={n_jobs}."
                )

            if hasattr(self._backend, "generate_spectrum_prefix_tree"):
                if verbose:
                    print("Using dense prefix-tree pathway reuse.")
                pathway_spectra = self._backend.generate_spectrum_prefix_tree(
                    protocol,
                    axis_values,
                    fixed_coordinates,
                    selected,
                    integration_weights,
                    parallel_backend=parallel_backend,
                    n_jobs=n_jobs,
                    block_size=block_size,
                    blas_threads=blas_threads,
                )
            else:
                pathway_spectra = {
                    pathway.name: np.zeros(shape, dtype=np.complex128)
                    for pathway in selected
                }
                for first_index, first_value in enumerate(axis_values[0]):
                    for second_index, second_value in enumerate(axis_values[1]):
                        coordinates = {
                            **fixed_coordinates,
                            axis_names[0]: first_value,
                            axis_names[1]: second_value,
                        }
                        for pathway in selected:
                            response = self.calc_pathway(
                                pathway, protocol, coordinates
                            )
                            pathway_spectra[pathway.name][
                                first_index, second_index
                            ] = self._integrate_k_response(
                                response, integration_weights
                            )

            components = {}
            for pathway in selected:
                if pathway.component not in components:
                    components[pathway.component] = np.zeros(
                        shape, dtype=np.complex128
                    )
                components[pathway.component] += pathway_spectra[pathway.name]
            return SpectrumResult(
                axis_names=axis_names,
                axis_values=axis_values,
                pathways=pathway_spectra,
                components=components,
                coherence_orders={
                    pathway.name: pathway.coherence_orders
                    for pathway in selected
                },
                fixed_coordinates=fixed_coordinates,
                pathway_metadata={
                    pathway.name: pathway.metadata()
                    for pathway in selected
                },
            )

    def generate_NQ_spectrum(
            self,
            order,
            protocol,
            axes,
            delays,
            pathways=None,
            k_array=None,
            verbose=True,
            k_weights=None,
            n_jobs=None,
            parallel_backend=None,
            block_size=None,
            blas_threads=None,
        ):
            """Generate and separate the positive/negative NQ contributions."""
            order = abs(int(order))
            nq_indices = [
                index
                for index, interval in enumerate(protocol.intervals)
                if interval.domain == "frequency"
                and interval.coherence_order is not None
                and abs(interval.coherence_order) == order
            ]
            if len(nq_indices) != 1:
                raise ValueError(
                    "The protocol must identify exactly one frequency interval "
                    f"with coherence order {order}."
                )
            nq_index = nq_indices[0]
            candidates = (
                list(self.pathways)
                if pathways is None
                else [self._resolve_pathway(item) for item in pathways]
            )
            selected = [
                pathway
                for pathway in candidates
                if pathway.coherence_orders
                and abs(pathway.coherence_orders[nq_index]) == order
            ]
            if not selected:
                raise ValueError(f"No pathways carry a {order}Q coherence.")
            result = self.generate_spectrum(
                protocol,
                axes,
                delays,
                pathways=selected,
                k_array=k_array,
                verbose=verbose,
                k_weights=k_weights,
                n_jobs=n_jobs,
                parallel_backend=parallel_backend,
                block_size=block_size,
                blas_threads=blas_threads,
            )
            shape = tuple(len(values) for values in result.axis_values)
            components = dict(result.components)
            total = np.zeros(shape, dtype=np.complex128)
            if order == 0:
                zero = np.zeros(shape, dtype=np.complex128)
                for pathway in selected:
                    zero += result.pathways[pathway.name]
                components["0Q"] = zero
                total = zero
            else:
                positive = np.zeros(shape, dtype=np.complex128)
                negative = np.zeros(shape, dtype=np.complex128)
                for pathway in selected:
                    target = (
                        positive
                        if pathway.coherence_orders[nq_index] > 0
                        else negative
                    )
                    target += result.pathways[pathway.name]
                components[f"+{order}Q"] = positive
                components[f"-{order}Q"] = negative
                total = positive + negative
            components[f"{order}Q"] = total
            return SpectrumResult(
                axis_names=result.axis_names,
                axis_values=result.axis_values,
                pathways=result.pathways,
                components=components,
                coherence_orders=result.coherence_orders,
                fixed_coordinates=result.fixed_coordinates,
                pathway_metadata=result.pathway_metadata,
            )

