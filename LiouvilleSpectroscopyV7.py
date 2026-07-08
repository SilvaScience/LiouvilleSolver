"""Dense Liouville-space solver for arbitrary-order impulsive spectroscopy.

The public API supports configurable initial states, UFSS pathway translation,
2D spectrum generation, individual pathway spectra, and pathway-grid plotting.
Only the dense Liouville backend is implemented. Backend selection remains an
explicit dispatch point so another implementation can be added independently.
"""

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


_PROCESS_SOLVER = None


def _init_process_solver(solver):
    """Store one solver copy per process worker."""
    global _PROCESS_SOLVER
    _PROCESS_SOLVER = solver


def _calc_w3_block_process(block, w_list, tau2, integration_weights, spectrum_components):
    """ProcessPool worker entry point using the process-local solver."""
    if _PROCESS_SOLVER is None:
        raise RuntimeError("Process worker was not initialized with a solver.")
    return _PROCESS_SOLVER._calc_w3_block(
        block, w_list, tau2, integration_weights, spectrum_components
    )


def _calc_w3_block_joblib(solver, block, w_list, tau2, integration_weights, spectrum_components):
    """Joblib worker entry point for process-style backends."""
    return solver._calc_w3_block(
        block, w_list, tau2, integration_weights, spectrum_components
    )


_COHERENCE_CHANGE = {"Ku": 1, "Kd": -1, "Bu": -1, "Bd": 1}


def _coherence_orders_from_interactions(interactions):
    """Return the signed coherence history implied by UFSS interactions."""
    current = 0
    orders = []
    for interaction in interactions:
        current += _COHERENCE_CHANGE[interaction]
        orders.append(current)
    return tuple(orders)


def _response_prefactor_from_interactions(interactions, amplitude=1.0):
    """Return the perturbative prefactor for an ordered Liouville pathway."""
    bra_sign = -1 if sum(item.startswith("B") for item in interactions) % 2 else 1
    return complex(amplitude) * (1j ** len(interactions)) * bra_sign


def _normalize_phase_discrimination(phase_discrimination):
    """Normalize a compact phase signature or UFSS phase tuples."""
    if isinstance(phase_discrimination, str):
        if not phase_discrimination:
            raise ValueError("phase_discrimination cannot be empty.")
        invalid = sorted(set(phase_discrimination).difference({"+", "-"}))
        if invalid:
            raise ValueError(
                "A phase signature may contain only '+' and '-'; "
                f"received invalid symbol(s) {invalid}."
            )
        return tuple(
            (1, 0) if sign == "+" else (0, 1)
            for sign in phase_discrimination
        )

    normalized = []
    for index, item in enumerate(phase_discrimination):
        if isinstance(item, str):
            if item not in {"+", "-"}:
                raise ValueError(
                    f"phase_discrimination[{index}] must be '+', '-', or "
                    "a two-entry UFSS tuple."
                )
            normalized.append((1, 0) if item == "+" else (0, 1))
            continue
        try:
            rotating, counter_rotating = item
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"phase_discrimination[{index}] must contain two entries."
            ) from exc
        rotating = int(rotating)
        counter_rotating = int(counter_rotating)
        if rotating < 0 or counter_rotating < 0:
            raise ValueError("Phase-discrimination counts must be non-negative.")
        normalized.append((rotating, counter_rotating))
    if not normalized:
        raise ValueError("phase_discrimination cannot be empty.")
    return tuple(normalized)


@dataclass(frozen=True)
class FrequencyPathway:
    """
    Frequency-domain representation of one UFSS density-matrix diagram.

    Parameters
    ----------
    name : str
        Public pathway label, for example ``R1``.
    interactions : tuple of str
        Ordered UFSS interaction labels (Ku, Kd, Bu, Bd).
    pulse_indices : tuple of int
        UFSS pulse index associated with each interaction.
    component : str
        Spectrum group, normally ``"rephasing"`` or ``"unrephasing"``.
    amplitude : complex
        Additional user-supplied pathway amplitude. The bra-side commutator
        sign is applied automatically for legacy third-order pathways.
    prefactor : complex or None
        Complete response prefactor. It is required outside third order.
    coherence_orders : tuple of int
        Signed coherence order after each interaction. When supplied, the
        complete history must match the interaction-implied history exactly.
    """

    name: str
    interactions: tuple
    pulse_indices: tuple = ()
    component: str = "custom"
    amplitude: complex = 1.0
    prefactor: complex = None
    coherence_orders: tuple = ()

    def __post_init__(self):
        interactions = tuple(str(item) for item in self.interactions)
        pulse_indices = tuple(int(item) for item in self.pulse_indices)
        coherence_orders = tuple(int(item) for item in self.coherence_orders)
        component = str(self.component).lower().replace("-", "")
        if component in {"nonrephasing", "nonrephase"}:
            component = "unrephasing"
        valid = {"Ku", "Kd", "Bu", "Bd"}
        unknown = [item for item in interactions if item not in valid]
        if unknown:
            raise ValueError(
                f"Unknown UFSS interaction label(s): {unknown}. "
                "Expected Ku, Kd, Bu, or Bd."
            )
        if pulse_indices and len(pulse_indices) != len(interactions):
            raise ValueError(
                "pulse_indices must be empty or have one entry per interaction."
            )
        if coherence_orders and len(coherence_orders) != len(interactions):
            raise ValueError(
                "coherence_orders must be empty or have one entry per interaction."
            )
        expected_coherence_orders = _coherence_orders_from_interactions(interactions)
        if coherence_orders and coherence_orders != expected_coherence_orders:
            mismatch_index = next(
                index
                for index, (declared, expected) in enumerate(
                    zip(coherence_orders, expected_coherence_orders), start=1
                )
                if declared != expected
            )
            raise ValueError(
                f"Signed coherence metadata mismatch after interaction "
                f"{mismatch_index} for pathway {self.name!r}: declared "
                f"q={coherence_orders[mismatch_index - 1]}, but interactions "
                f"imply q={expected_coherence_orders[mismatch_index - 1]}."
            )
        if self.prefactor is None and len(interactions) != 3:
            raise ValueError(
                "An explicit prefactor is required for pathways whose order "
                "is not three."
            )
        object.__setattr__(self, "interactions", interactions)
        object.__setattr__(self, "pulse_indices", pulse_indices)
        object.__setattr__(self, "coherence_orders", coherence_orders)
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "amplitude", complex(self.amplitude))
        if self.prefactor is not None:
            object.__setattr__(self, "prefactor", complex(self.prefactor))

    @property
    def bra_sign(self):
        """Commutator sign generated by bra-side interactions."""
        return -1 if sum(item.startswith("B") for item in self.interactions) % 2 else 1

    @property
    def coefficient(self):
        """Total relative pathway coefficient, excluding the common ``-1j``."""
        return self.amplitude * self.bra_sign

    @property
    def ufss_diagram(self):
        """Return the ordered UFSS instructions represented by this pathway."""
        pulse_indices = (
            self.pulse_indices
            if self.pulse_indices
            else tuple(range(len(self.interactions)))
        )
        return tuple(zip(self.interactions, pulse_indices))

    def metadata(self):
        """Return serializable plotting and provenance metadata."""
        return {
            "name": self.name,
            "interactions": self.interactions,
            "pulse_indices": self.pulse_indices,
            "coherence_orders": self.coherence_orders,
            "component": self.component,
            "amplitude": self.amplitude,
            "prefactor": self.prefactor,
            "response_prefactor": self.response_prefactor,
            "ufss_diagram": self.ufss_diagram,
        }

    @property
    def response_prefactor(self):
        """Return the complete complex response prefactor."""
        if self.prefactor is not None:
            return self.prefactor
        return -1j * self.coefficient


@dataclass(frozen=True)
class PropagationInterval:
    """One propagation interval following a light-matter interaction."""

    name: str
    domain: str
    coherence_order: int = None

    def __post_init__(self):
        domain = str(self.domain).lower()
        if domain not in {"frequency", "time", "identity"}:
            raise ValueError("domain must be 'frequency', 'time', or 'identity'.")
        name = str(self.name)
        if domain != "identity" and not name:
            raise ValueError("Frequency and time intervals require a name.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "domain", domain)
        if self.coherence_order is not None:
            object.__setattr__(self, "coherence_order", int(self.coherence_order))


@dataclass(frozen=True)
class SpectroscopyProtocol:
    """Propagation topology used to evaluate a pathway."""

    intervals: tuple
    name: str = "custom"

    def __post_init__(self):
        intervals = tuple(
            item if isinstance(item, PropagationInterval)
            else PropagationInterval(**item)
            for item in self.intervals
        )
        if not intervals:
            raise ValueError("A protocol requires at least one interval.")
        names = [item.name for item in intervals if item.domain != "identity"]
        if len(names) != len(set(names)):
            raise ValueError("Non-identity interval names must be unique.")
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "name", str(self.name))

    @property
    def frequency_axis_names(self):
        return tuple(
            item.name for item in self.intervals if item.domain == "frequency"
        )

    @property
    def time_interval_names(self):
        return tuple(item.name for item in self.intervals if item.domain == "time")

    def validate_pathway(self, pathway):
        if len(pathway.interactions) != len(self.intervals):
            raise ValueError(
                f"Protocol {self.name!r} has {len(self.intervals)} intervals, "
                f"but pathway {pathway.name!r} has "
                f"{len(pathway.interactions)} interactions."
            )
        if pathway.coherence_orders:
            for index, (expected, actual) in enumerate(
                zip(self.intervals, pathway.coherence_orders), start=1
            ):
                if (
                    expected.coherence_order is not None
                    and abs(actual) != abs(expected.coherence_order)
                ):
                    raise ValueError(
                        f"Coherence mismatch after interaction {index}: "
                        f"protocol expects |q|={abs(expected.coherence_order)}, "
                        f"pathway {pathway.name!r} has q={actual}."
                    )


@dataclass(frozen=True)
class SpectrumResult:
    """Two-dimensional spectra and their axis/pathway metadata."""

    axis_names: tuple
    axis_values: tuple
    pathways: dict
    components: dict
    coherence_orders: dict
    fixed_coordinates: dict
    pathway_metadata: dict = None

    def __post_init__(self):
        if self.pathway_metadata is None:
            self.pathway_metadata = {}


@dataclass
class PathwayPlotResult:
    """Matplotlib and UFSS artifacts produced by a multi-order pathway plot."""

    figure: object
    axes: object
    panel_names: tuple
    diagrams: dict
    diagram_paths: dict
    spectrum_pdf: object = None


def standard_nq_protocol(
    order,
    nq_interval,
    detection_interval,
    n_interactions=3,
    nq_axis=None,
    detection_axis="omega3",
):
    """Build a two-frequency protocol; interval numbers are one-based."""
    order = abs(int(order))
    n_interactions = int(n_interactions)
    nq_index = int(nq_interval) - 1
    detection_index = int(detection_interval) - 1
    if n_interactions < 1:
        raise ValueError("n_interactions must be positive.")
    if not (0 <= nq_index < n_interactions):
        raise ValueError("nq_interval is outside the protocol.")
    if not (0 <= detection_index < n_interactions):
        raise ValueError("detection_interval is outside the protocol.")
    if nq_index == detection_index:
        raise ValueError("The NQ and detection intervals must be different.")
    nq_axis = nq_axis or f"omega{order}q"
    intervals = []
    for index in range(n_interactions):
        if index == nq_index:
            intervals.append(PropagationInterval(nq_axis, "frequency", order))
        elif index == detection_index:
            intervals.append(PropagationInterval(detection_axis, "frequency"))
        else:
            intervals.append(PropagationInterval(f"t{index + 1}", "time"))
    return SpectroscopyProtocol(tuple(intervals), name=f"standard_{order}q")


def standard_1q_protocol():
    return standard_nq_protocol(1, 1, 3, nq_axis="omega1")


def standard_2q_protocol():
    return standard_nq_protocol(2, 2, 3, nq_axis="omega2q")


def standard_0q_protocol():
    return standard_nq_protocol(0, 2, 3, nq_axis="omega0q")
###################################################################################
#################             Author: Mathieu Desmarais            ################
#################                Date: 25-06-2026                  ################
#################           2D spectroscopy solver - V6            ################
###################################################################################


class LiouvilleSpectroscopySolver:
    _BACKEND_BUILDERS = {"dense": "_build_dense_liouville"}
    _BACKEND_PATHWAY_CALCULATORS = {"dense": "_calc_pathway_dense"}

    def __init__(self, params):
        """
        Universal Liouville-space solver for 2D spectroscopy.

        Parameters
        ----------
        params : dict
            Expected keys include:
            - Eta: spectral broadening
            - T: temperature
            - mu: chemical potential
            - cache_resolvents: whether dense resolvent matrices are cached
            - max_resolvent_cache: optional maximum number of cached factorizations
        """
        self.params = params
        self.eta = params.get("Eta", 0.05)
        self.T = params.get("T", 0.01)
        self.rwa_tol = params.get("rwa_tol", 1e-6)
        self.cache_resolvents = params.get("cache_resolvents", True)
        self.max_resolvent_cache = params.get("max_resolvent_cache", None)
        self.backend = str(params.get("backend", "dense")).lower()
        self.parallel_backend = params.get("parallel_backend", "threading")
        self.parallel_block_size = params.get("parallel_block_size", None)
        self.blas_threads = params.get("blas_threads", None)
        self.n_jobs = params.get("n_jobs", -1)
        self.spectrum_components = self._normalize_spectrum_components(
            params.get("spectrum_components", "both")
        )
        self.density_matrix_tolerance = params.get(
            "density_matrix_tolerance", 1e-10
        )
        self._active_backend = None

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
        self.pathways = self._default_frequency_pathways()

    def _normalize_spectrum_components(self, spectrum_components):
        """Normalize the requested 2D spectrum components."""
        if spectrum_components is None:
            spectrum_components = getattr(self, "spectrum_components", "both")
        spectrum_components = str(spectrum_components).lower()
        valid = {"both", "rephasing", "unrephasing"}
        if spectrum_components not in valid:
            raise ValueError(
                "spectrum_components must be one of: "
                "'both', 'rephasing', or 'unrephasing'"
            )
        return spectrum_components

    def _wants_rephasing(self, spectrum_components):
        return spectrum_components in {"both", "rephasing"}

    def _wants_unrephasing(self, spectrum_components):
        return spectrum_components in {"both", "unrephasing"}

    def _format_spectra_result(self, S3_reph, S3_unreph, spectrum_components):
        """Build the public spectra dictionary for the requested components."""
        result = {}
        if self._wants_rephasing(spectrum_components):
            result["rephasing"] = S3_reph
        if self._wants_unrephasing(spectrum_components):
            result["unrephasing"] = S3_unreph
        if spectrum_components == "both":
            result["absorptive"] = S3_reph + S3_unreph
        return result

    # ========================================================================
    # PATHWAY DEFINITIONS AND UFSS TRANSLATION
    # ========================================================================

    def _default_frequency_pathways(self):
        """
        Return the six impulsive third-order pathways used by the old solver.

        The R labels follow the ordering used by UFSS/QuDPy examples:
        R1=SE, R2=GSB, R3=ESA for rephasing and R4=ESA, R5=SE,
        R6=GSB for unrephasing.
        """
        return [
            FrequencyPathway(
                "R1", ("Bu", "Ku", "Bd"), (0, 1, 2), "rephasing",
                coherence_orders=(-1, 0, 1),
            ),
            FrequencyPathway(
                "R2", ("Bu", "Bd", "Ku"), (0, 1, 2), "rephasing",
                coherence_orders=(-1, 0, 1),
            ),
            FrequencyPathway(
                "R3", ("Bu", "Ku", "Ku"), (0, 1, 2), "rephasing",
                coherence_orders=(-1, 0, 1),
            ),
            FrequencyPathway(
                "R4", ("Ku", "Bu", "Ku"), (0, 1, 2), "unrephasing",
                coherence_orders=(1, 0, 1),
            ),
            FrequencyPathway(
                "R5", ("Ku", "Bu", "Bd"), (0, 1, 2), "unrephasing",
                coherence_orders=(1, 0, 1),
            ),
            FrequencyPathway(
                "R6", ("Ku", "Kd", "Ku"), (0, 1, 2), "unrephasing",
                coherence_orders=(1, 0, 1),
            ),
        ]

    def reset_pathways(self):
        """Restore the six standard third-order pathways."""
        self.pathways = self._default_frequency_pathways()

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

    def _phase_discrimination_component(self, phase_discrimination):
        """Infer the standard 2D component from a UFSS phase condition."""
        normalized = tuple(
            "+" * rotating + "-" * counter_rotating
            for rotating, counter_rotating in _normalize_phase_discrimination(
                phase_discrimination
            )
        )
        if normalized == ("-", "+", "+"):
            return "rephasing"
        if normalized == ("+", "-", "+"):
            return "unrephasing"
        return "custom"

    def _make_pathway_names_unique(self, pathways):
        """Return pathways with stable unique names while preserving metadata."""
        unique = []
        counts = {}
        for pathway in pathways:
            base_name = pathway.name
            counts[base_name] = counts.get(base_name, 0) + 1
            name = (
                base_name
                if counts[base_name] == 1
                else f"{base_name}_{counts[base_name]}"
            )
            if name == pathway.name:
                unique.append(pathway)
            else:
                unique.append(
                    FrequencyPathway(
                        name=name,
                        interactions=pathway.interactions,
                        pulse_indices=pathway.pulse_indices,
                        component=pathway.component,
                        amplitude=pathway.amplitude,
                        prefactor=pathway.prefactor,
                        coherence_orders=pathway.coherence_orders,
                    )
                )
        return unique

    def translate_ufss_diagrams(
        self,
        diagrams,
        component="auto",
        names=None,
        amplitudes=None,
        prefactors=None,
        coherence_orders=None,
        allow_noncanonical_order=False,
    ):
        """
        Translate UFSS DiagramGenerator output to frequency pathways.

        UFSS diagrams use ordered ``(instruction, pulse_index)`` pairs. This
        translator preserves that ordering and converts the four UFSS
        instructions directly to the corresponding left/right Liouville
        superoperators. Arbitrary pathway orders are supported, with
        non-decreasing pulse indices required by default.
        """
        diagrams = list(diagrams)
        if names is None:
            names = [None] * len(diagrams)
        if amplitudes is None:
            amplitudes = [1.0] * len(diagrams)
        if prefactors is None:
            prefactors = [None] * len(diagrams)
        if coherence_orders is None:
            coherence_orders = [None] * len(diagrams)
        metadata = (names, amplitudes, prefactors, coherence_orders)
        if any(len(items) != len(diagrams) for items in metadata):
            raise ValueError(
                "names, amplitudes, prefactors, and coherence_orders must "
                "have one entry per UFSS diagram."
            )

        standard_names = {
            ("Bu", "Ku", "Bd"): ("R1", "rephasing"),
            ("Bu", "Bd", "Ku"): ("R2", "rephasing"),
            ("Bu", "Ku", "Ku"): ("R3", "rephasing"),
            ("Ku", "Bu", "Ku"): ("R4", "unrephasing"),
            ("Ku", "Bu", "Bd"): ("R5", "unrephasing"),
            ("Ku", "Kd", "Ku"): ("R6", "unrephasing"),
        }
        standard_coherences = {
            "R1": (-1, 0, 1),
            "R2": (-1, 0, 1),
            "R3": (-1, 0, 1),
            "R4": (1, 0, 1),
            "R5": (1, 0, 1),
            "R6": (1, 0, 1),
        }

        translated = []
        used_names = set()
        requested_component = (
            None if component is None else str(component).lower()
        )
        for index, (
            diagram,
            name,
            amplitude,
            prefactor,
            pathway_coherences,
        ) in enumerate(
            zip(
                diagrams,
                names,
                amplitudes,
                prefactors,
                coherence_orders,
            ),
            start=1,
        ):
            diagram = tuple(diagram)
            if not diagram:
                raise ValueError(f"UFSS diagram {index} is empty.")
            try:
                interactions = tuple(item[0] for item in diagram)
                pulse_indices = tuple(item[1] for item in diagram)
            except (TypeError, IndexError) as exc:
                raise ValueError(
                    "Each UFSS interaction must be an "
                    "(instruction, pulse_index) pair."
                ) from exc
            if pathway_coherences is None or len(pathway_coherences) == 0:
                pathway_coherences = _coherence_orders_from_interactions(
                    interactions
                )
            if prefactor is None and len(interactions) != 3:
                prefactor = _response_prefactor_from_interactions(
                    interactions, amplitude=amplitude
                )
            is_time_ordered = all(
                earlier <= later
                for earlier, later in zip(
                    pulse_indices, pulse_indices[1:]
                )
            )
            if not is_time_ordered:
                message = (
                    f"UFSS diagram {index} has pulse order {pulse_indices}, "
                    "which is not non-decreasing."
                )
                if not allow_noncanonical_order:
                    raise ValueError(
                        message
                        + " Use non-overlapping, time-ordered pulses or set "
                        "allow_noncanonical_order=True."
                    )
                warnings.warn(
                    message + " The supplied interaction order is preserved.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            if len(diagram) == 3 and pulse_indices == (0, 1, 2):
                standard_name, standard_component = standard_names.get(
                    interactions, (None, None)
                )
            else:
                # R1-R6 labels describe the canonical pulse ordering only.
                standard_name, standard_component = (None, None)
            pathway_name = name or standard_name or f"P{index}"
            base_name = pathway_name
            suffix = 2
            while pathway_name in used_names:
                pathway_name = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(pathway_name)
            pathway_component = (
                standard_component
                if requested_component in {None, "auto"}
                else requested_component
            )
            if pathway_component is None:
                pathway_component = "custom"
            if standard_name is not None:
                expected_standard = standard_coherences[standard_name]
                if pathway_coherences != expected_standard:
                    raise ValueError(
                        f"UFSS pathway {pathway_name!r} has coherence history "
                        f"{pathway_coherences}, expected {expected_standard}."
                    )
            translated.append(
                FrequencyPathway(
                    name=pathway_name,
                    interactions=interactions,
                    pulse_indices=pulse_indices,
                    component=pathway_component,
                    amplitude=amplitude,
                    prefactor=prefactor,
                    coherence_orders=pathway_coherences,
                )
            )
        return translated

    def generate_pathways_with_ufss(
        self,
        phase_discrimination,
        arrival_times=None,
        component="auto",
        efield_times=None,
        names=None,
        amplitudes=None,
        prefactors=None,
        coherence_orders=None,
        detection_type="polarization",
        maximum_manifold=None,
        replace=False,
        allow_noncanonical_order=False,
    ):
        """
        Generate diagrams with UFSS and translate them for this solver.

        ``phase_discrimination`` accepts either UFSS tuples or a compact string
        such as ``"++--+"``. If ``arrival_times`` is omitted, consecutive unit
        arrival times are used. If ``efield_times`` is omitted, delta-like
        pulses are used. Finite pulse windows may cause UFSS to return reordered
        interactions. Those diagrams are rejected unless
        ``allow_noncanonical_order=True`` because this solver does not perform
        finite-pulse envelope convolutions.

        Coherence histories are inferred from the UFSS instructions. Complete
        perturbative prefactors are inferred automatically outside third order;
        explicit metadata remains available as an override. Set
        ``maximum_manifold`` whenever the UFSS default does not match the model.
        """
        try:
            import ufss
        except ImportError as exc:
            raise ImportError(
                "UFSS is required for pathway generation. Install it with "
                "`pip install ufss`."
            ) from exc

        phase_discrimination = _normalize_phase_discrimination(
            phase_discrimination
        )
        n_pulses = len(phase_discrimination)
        if arrival_times is None:
            arrival_times = np.arange(n_pulses, dtype=float)
        else:
            arrival_times = np.asarray(arrival_times, dtype=float)
        if arrival_times.ndim != 1 or len(arrival_times) != n_pulses:
            raise ValueError(
                "arrival_times must be one-dimensional with one value per "
                f"phase-discrimination entry; expected {n_pulses}."
            )

        generator = ufss.DiagramGenerator(detection_type=detection_type)
        generator.set_phase_discrimination(phase_discrimination)
        if maximum_manifold is not None:
            maximum_manifold = int(maximum_manifold)
            if maximum_manifold < 0:
                raise ValueError("maximum_manifold must be non-negative.")
            generator.maximum_manifold = maximum_manifold
        if efield_times is None:
            generator.efield_times = [
                np.array([0.0, 0.0], dtype=float)
                for _ in range(n_pulses)
            ]
        else:
            generator.efield_times = [
                np.asarray(times, dtype=float) for times in efield_times
            ]
            if len(generator.efield_times) != n_pulses:
                raise ValueError(
                    "efield_times must contain one window per "
                    f"phase-discrimination entry; expected {n_pulses}."
                )

        diagrams = generator.get_diagrams(arrival_times)
        translated_component = component
        if component is None or str(component).lower() == "auto":
            translated_component = self._phase_discrimination_component(
                phase_discrimination
            )
        pathways = self.translate_ufss_diagrams(
            diagrams,
            component=translated_component,
            names=names,
            amplitudes=amplitudes,
            prefactors=prefactors,
            coherence_orders=coherence_orders,
            allow_noncanonical_order=allow_noncanonical_order,
        )
        if replace:
            self.set_pathways(pathways)
        return pathways

    def set_pathways_from_ufss(
        self,
        diagrams,
        component="auto",
        names=None,
        amplitudes=None,
        prefactors=None,
        coherence_orders=None,
        allow_noncanonical_order=False,
    ):
        """Translate UFSS output and immediately make it active."""
        pathways = self.translate_ufss_diagrams(
            diagrams,
            component=component,
            names=names,
            amplitudes=amplitudes,
            prefactors=prefactors,
            coherence_orders=coherence_orders,
            allow_noncanonical_order=allow_noncanonical_order,
        )
        self.set_pathways(pathways)
        return pathways

    def configure_standard_2d_pathways_with_ufss(
        self,
        arrival_times,
        efield_times=None,
        allow_noncanonical_order=False,
    ):
        """
        Ask UFSS for both standard third-order 2D phase-matching groups.

        The resulting active list is ordered R1 through R6 when all six
        standard impulsive diagrams are present.
        """
        rephasing = self.generate_pathways_with_ufss(
            [(0, 1), (1, 0), (1, 0)],
            arrival_times,
            component="auto",
            efield_times=efield_times,
            allow_noncanonical_order=allow_noncanonical_order,
        )
        unrephasing = self.generate_pathways_with_ufss(
            [(1, 0), (0, 1), (1, 0)],
            arrival_times,
            component="auto",
            efield_times=efield_times,
            allow_noncanonical_order=allow_noncanonical_order,
        )
        pathways = self._make_pathway_names_unique(
            rephasing + unrephasing
        )
        pathways.sort(
            key=lambda item: (
                0,
                int(item.name[1:]),
            )
            if item.name.startswith("R") and item.name[1:].isdigit()
            else (1, item.name)
        )
        self.set_pathways(pathways)
        return pathways

    def _using_default_pathways(self):
        """Whether optimized hard-coded contractions remain applicable."""
        return self.pathways == self._default_frequency_pathways()

    # ========================================================================
    # Parallelisation method
    # ========================================================================

    def _calc_w3_column(
        self, i, w3, w_list, tau2, integration_weights, spectrum_components
    ):
        """
        Calculate a complete omega column of the 2D spectra for a fixed w3.
        This function is executed by an independent worker.
        """
        n_w = len(w_list)
        col_reph = (
            np.zeros(n_w, dtype=np.complex128)
            if self._wants_rephasing(spectrum_components)
            else None
        )
        col_unreph = (
            np.zeros(n_w, dtype=np.complex128)
            if self._wants_unrephasing(spectrum_components)
            else None
        )

        for j, w1 in enumerate(w_list):
            if col_reph is not None:
                vect_reph = self.calc_rephasing(w3, w1, tau2)
                col_reph[j] = self._integrate_k_response(
                    vect_reph, integration_weights
                )
            if col_unreph is not None:
                vect_unreph = self.calc_unrephasing(w3, w1, tau2)
                col_unreph[j] = self._integrate_k_response(
                    vect_unreph, integration_weights
                )

        return i, col_reph, col_unreph

    def _calc_w3_block(
        self, block, w_list, tau2, integration_weights, spectrum_components
    ):
        """Calculate a block of omega_3 columns."""
        return [
            self._calc_w3_column(
                i, w_list[i], w_list, tau2, integration_weights, spectrum_components
            )
            for i in block
        ]

    def _effective_n_jobs(self, n_jobs):
        """Normalize joblib-style worker counts."""
        cpu_count = os.cpu_count() or 1
        if n_jobs is None:
            return 1
        if n_jobs < 0:
            return max(1, cpu_count + 1 + n_jobs)
        return max(1, int(n_jobs))

    def _make_w3_blocks(self, n_w, n_jobs, block_size):
        """Split omega_3 column indices into backend tasks."""
        if block_size is None:
            block_size = self.parallel_block_size
        if block_size is None:
            target_blocks = max(1, 4 * max(1, n_jobs))
            block_size = max(1, math.ceil(n_w / target_blocks))
        block_size = max(1, int(block_size))
        return [
            list(range(start, min(start + block_size, n_w)))
            for start in range(0, n_w, block_size)
        ]

    def _parallel_context(self, blas_threads):
        """Limit nested BLAS/OpenMP threads while the outer omega loop runs."""
        if blas_threads is None:
            blas_threads = self.blas_threads
        if blas_threads is None or threadpool_limits is None:
            return nullcontext()
        return threadpool_limits(limits=blas_threads)

    def _run_w3_blocks(
        self,
        blocks,
        w_list,
        tau2,
        integration_weights,
        parallel_backend,
        n_jobs,
        spectrum_components,
    ):
        """Run omega_3 blocks using the selected parallel backend."""
        if parallel_backend in {"serial", None} or n_jobs == 1:
            return [
                item
                for block in blocks
                for item in self._calc_w3_block(
                    block, w_list, tau2, integration_weights, spectrum_components
                )
            ]

        if parallel_backend == "threading":
            nested = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(self._calc_w3_block)(
                    block, w_list, tau2, integration_weights, spectrum_components
                )
                for block in blocks
            )
        elif parallel_backend in {"loky", "multiprocessing"}:
            nested = Parallel(
                n_jobs=n_jobs,
                backend=parallel_backend,
                max_nbytes="10M",
                mmap_mode="r",
            )(
                delayed(_calc_w3_block_joblib)(
                    self,
                    block,
                    w_list,
                    tau2,
                    integration_weights,
                    spectrum_components,
                )
                for block in blocks
            )
        elif parallel_backend in {"process", "processpool"}:
            with ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=_init_process_solver,
                initargs=(self,),
            ) as pool:
                futures = [
                    pool.submit(
                        _calc_w3_block_process,
                        block,
                        w_list,
                        tau2,
                        integration_weights,
                        spectrum_components,
                    )
                    for block in blocks
                ]
                nested = [future.result() for future in futures]
        else:
            raise ValueError(
                "parallel_backend must be one of: "
                "'serial', 'threading', 'loky', 'multiprocessing', or 'process'"
            )

        return [item for block_result in nested for item in block_result]


    # ========================================================================
    # INPUT NORMALIZATION
    # ========================================================================


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

    # ========================================================================
    # INITIAL DENSITY MATRIX
    # ========================================================================

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
        self._rho_eq_dense = self._get_initial_state()

    # ========================================================================
    # MODEL LOADING
    # ========================================================================
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

    # ========================================================================
    # BACKEND DISPATCH
    # ========================================================================
    def _select_backend(self):
        """Validate and return the requested Liouville backend."""
        if self.backend not in self._BACKEND_BUILDERS:
            supported = ", ".join(sorted(self._BACKEND_BUILDERS))
            raise ValueError(
                f"Unsupported Liouville backend {self.backend!r}. "
                f"Available backend(s): {supported}."
            )
        return self.backend

    def _build_liouville_backend(self):
        """Build the selected backend through the backend registry."""
        self._active_backend = self._select_backend()
        self._dense_resolvent_cache.clear()
        self._dense_time_cache.clear()
        builder = getattr(self, self._BACKEND_BUILDERS[self._active_backend])
        builder()

    # ========================================================================
    # DENSE VECTORIZED LIOUVILLE ALGEBRA
    # ========================================================================
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
        cached = self._dense_resolvent_cache.get(key)
        if cached is not None:
            return cached

   
        A = (w + 1j * self.eta) * self._I_super_dense - self._L_eff_dense
        G = np.linalg.inv(A)
      

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

    def _calc_rephasing_dense(self, w3, w1, tau2):
        """Compute rephasing diagrams with the dense batched backend."""
      

        G1 = self._get_dense_resolvent(w1)
        G2 = self._get_dense_time_propagator(tau2)
        G3 = self._get_dense_resolvent(w3)
        rho = self._rho_eq_dense

        path_gsb = G3 @ (
            self._JL_plus_dense
            @ (G2 @ (self._JR_plus_dense @ (G1 @ (self._JR_minus_dense @ rho))))
        )
        path_se = G3 @ (
            self._JR_plus_dense
            @ (G2 @ (self._JL_plus_dense @ (G1 @ (self._JR_minus_dense @ rho))))
        )
        path_esa = G3 @ (
            self._JL_plus_dense
            @ (G2 @ (self._JL_plus_dense @ (G1 @ (self._JR_minus_dense @ rho))))
        )

        tr_gsb = (self._trace_vec_dense @ (self._JL_out_dense @ path_gsb)).reshape(-1)
        tr_se = (self._trace_vec_dense @ (self._JL_out_dense @ path_se)).reshape(-1)
        tr_esa = (self._trace_vec_dense @ (self._JL_out_dense @ path_esa)).reshape(-1)

       
        return -1j * (tr_gsb + tr_se - tr_esa)

    def _calc_unrephasing_dense(self, w3, w1, tau2):
        """Compute non-rephasing diagrams with the dense batched backend."""
    

        G1 = self._get_dense_resolvent(w1)
        G2 = self._get_dense_time_propagator(tau2)
        G3 = self._get_dense_resolvent(w3)
        rho = self._rho_eq_dense

        path_gsb = G3 @ (
            self._JL_plus_dense
            @ (G2 @ (self._JL_minus_dense @ (G1 @ (self._JL_plus_dense @ rho))))
        )
        path_se = G3 @ (
            self._JR_plus_dense
            @ (G2 @ (self._JR_minus_dense @ (G1 @ (self._JL_plus_dense @ rho))))
        )
        path_esa = G3 @ (
            self._JL_plus_dense
            @ (G2 @ (self._JR_minus_dense @ (G1 @ (self._JL_plus_dense @ rho))))
        )

        tr_gsb = (self._trace_vec_dense @ (self._JL_out_dense @ path_gsb)).reshape(-1)
        tr_se = (self._trace_vec_dense @ (self._JL_out_dense @ path_se)).reshape(-1)
        tr_esa = (self._trace_vec_dense @ (self._JL_out_dense @ path_esa)).reshape(-1)

       
        return -1j * (tr_gsb + tr_se - tr_esa)

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
            column = -1j * self._integrate_k_response(
                traces[0] + traces[1] - traces[2],
                integration_weights,
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
        blocks = self._make_w3_blocks(n_w, n_jobs, block_size)

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

    def _generate_2D_spectra_dense(
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

    # ========================================================================
    # RESPONSE FUNCTIONS
    # ========================================================================
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

    def _interaction_dense(self, instruction):
        """Return the dense superoperator for one UFSS instruction."""
        return {
            "Ku": self._JL_plus_dense,
            "Kd": self._JL_minus_dense,
            "Bu": self._JR_minus_dense,
            "Bd": self._JR_plus_dense,
        }[instruction]

    def _calc_pathway_dense(self, pathway, protocol, coordinates):
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
            if interval.domain == "frequency":
                response = (
                    self._get_dense_resolvent(float(coordinates[interval.name]))
                    @ response
                )
            elif interval.domain == "time":
                response = (
                    self._get_dense_time_propagator(
                        float(coordinates[interval.name])
                    )
                    @ response
                )
        traces = (
            self._trace_vec_dense @ (self._JL_out_dense @ response)
        ).reshape(-1)
        return pathway.response_prefactor * traces

    def calc_pathway(
        self,
        pathway,
        protocol_or_w3,
        coordinates_or_w1=None,
        tau2=None,
    ):
        """
        Evaluate a pathway using a protocol or the legacy third-order call.

        The protocol form takes a SpectroscopyProtocol and coordinate mapping.
        The legacy form takes w3, w1, and tau2.
        """
        if self._active_backend is None:
            raise RuntimeError("Call feed_model() before calc_pathway().")
        pathway = self._resolve_pathway(pathway)
        if isinstance(protocol_or_w3, SpectroscopyProtocol):
            protocol = protocol_or_w3
            if coordinates_or_w1 is None:
                raise TypeError("coordinates are required with a protocol.")
            coordinates = dict(coordinates_or_w1)
        else:
            if coordinates_or_w1 is None or tau2 is None:
                raise TypeError(
                    "Legacy calc_pathway requires w3, w1, and tau2."
                )
            protocol = standard_1q_protocol()
            coordinates = {
                "omega1": float(coordinates_or_w1),
                "t2": float(tau2),
                "omega3": float(protocol_or_w3),
            }

        calculator_name = self._BACKEND_PATHWAY_CALCULATORS[self._active_backend]
        calculator = getattr(self, calculator_name)
        return calculator(pathway, protocol, coordinates)

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

    def calc_rephasing(self, w3, w1, tau2):
        """Compute rephasing diagrams (-k1, +k2, +k3)."""
        pathways = self.get_pathways("rephasing")
        response = np.zeros(self.N_k, dtype=np.complex128)
        for pathway in pathways:
            response += self.calc_pathway(pathway, w3, w1, tau2)
        return response

    def calc_unrephasing(self, w3, w1, tau2):
        """Compute non-rephasing diagrams (+k1, -k2, +k3)."""
        pathways = self.get_pathways("unrephasing")
        response = np.zeros(self.N_k, dtype=np.complex128)
        for pathway in pathways:
            response += self.calc_pathway(pathway, w3, w1, tau2)
        return response

    def _resolve_k_weights(self, k_array=None, k_weights=None):
        """Return validated per-k integration weights.

        Explicit ``k_weights`` are used as supplied and must already include
        the desired integration measure, such as ``dk / (2*pi)``.  When they
        are omitted, the historical scalar rule derived from ``k_array`` is
        retained for backward compatibility.  A model without either input
        keeps the historical unweighted sum over k.
        """
        if self.N_k is None:
            raise RuntimeError("Call feed_model() before resolving k weights.")

        if k_array is not None:
            k_array = np.asarray(k_array, dtype=float)
            if k_array.ndim != 1 or len(k_array) != self.N_k:
                raise ValueError(
                    "k_array must be one-dimensional with length "
                    f"{self.N_k}; got shape {k_array.shape}"
                )
            if not np.all(np.isfinite(k_array)):
                raise ValueError("k_array must contain only finite values")

        if k_weights is not None:
            raw_weights = np.asarray(k_weights)
            if np.iscomplexobj(raw_weights):
                raise ValueError("k_weights must be real")
            weights = np.asarray(raw_weights, dtype=float)
            if weights.ndim != 1 or len(weights) != self.N_k:
                raise ValueError(
                    "k_weights must be one-dimensional with length "
                    f"{self.N_k}; got shape {weights.shape}"
                )
            if not np.all(np.isfinite(weights)):
                raise ValueError("k_weights must contain only finite values")
            return weights.copy()

        if k_array is None or self.N_k == 1:
            return np.ones(self.N_k, dtype=float)

        warnings.warn(
            "Inferring rectangular integration weights from k_array uses the "
            "legacy endpoint convention. Pass explicit k_weights for periodic, "
            "trapezoidal, nonuniform, or multidimensional quadrature.",
            FutureWarning,
            stacklevel=3,
        )
        legacy_weight = (
            (k_array[-1] - k_array[0]) / self.N_k / (2 * np.pi)
        )
        return np.full(self.N_k, legacy_weight, dtype=float)

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
        return np.sum(response * np.asarray(k_weights).reshape(shape), axis=axis)

    def generate_spectrum(
        self,
        protocol,
        axes,
        delays,
        pathways=None,
        k_array=None,
        verbose=True,
        k_weights=None,
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
        pathway_spectra = {
            pathway.name: np.zeros(shape, dtype=np.complex128)
            for pathway in selected
        }
        self._dense_resolvent_cache.clear()
        self._dense_time_cache.clear()
        if verbose:
            print(
                f"Calculating {len(selected)} pathway spectrum/s on a "
                f"{shape[0]}x{shape[1]} grid with protocol {protocol.name!r}."
            )

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

    def generate_2D_spectra(
        self,
        w_list,
        tau2,
        k_array=None,
        n_jobs=None,
        parallel_backend=None,
        block_size=None,
        blas_threads=None,
        spectrum_components=None,
        verbose=True,
        k_weights=None,
    ):
        """
        Scan the w1/w3 grid and integrate over k.

        Parameters
        ----------
        k_array : array_like or None
            Legacy one-dimensional momentum samples. Used to infer the old
            rectangular factor only when ``k_weights`` is omitted.
        k_weights : array_like or None
            Explicit complete integration weights, one per k-point. These
            weights must already include the desired Brillouin-zone measure.
        parallel_backend : {"serial", "threading", "loky", "multiprocessing", "process"}
            Backend used for the omega_3 column loop. "process" uses a
            ProcessPoolExecutor initializer so the solver is copied once per
            worker instead of once per omega block.
        block_size : int or None
            Number of omega_3 columns per submitted task. Larger blocks reduce
            process-backend overhead.
        blas_threads : int or None
            Optional limit for nested BLAS/OpenMP threads during this scan.
        spectrum_components : {"both", "rephasing", "unrephasing"} or None
            Components to calculate. None uses the value configured in params.

        Returns
        -------
        dict
            Complex 2D response matrices for the requested components. The
            absorptive spectrum is returned only when both components are
            requested.
        """
        if n_jobs is None:
            n_jobs = self.n_jobs
        if self.N_k is None:
            raise RuntimeError("Call feed_model() before generate_2D_spectra().")

        spectrum_components = self._normalize_spectrum_components(
            spectrum_components
        )

        self._dense_resolvent_cache.clear()
        self._dense_time_cache.clear()

        w_list = np.asarray(w_list, dtype=float)
        n_w = len(w_list)

        integration_weights = self._resolve_k_weights(k_array, k_weights)

        if parallel_backend is None:
            parallel_backend = self.parallel_backend
        n_jobs_eff = self._effective_n_jobs(n_jobs)

        if verbose:
            print(
                f"Starting 2D scan on a {n_w}x{n_w} frequency grid "
                f"with Liouville={self._active_backend}, "
                f"components={spectrum_components}, "
                f"parallel={parallel_backend}, n_jobs={n_jobs_eff}."
            )

        if (
            self._active_backend == "dense"
            and self._using_default_pathways()
            and (
            parallel_backend in {"serial", "threading", None} or n_jobs_eff == 1
            )
        ):
            with self._parallel_context(blas_threads):
                return self._generate_2D_spectra_dense(
                    w_list,
                    tau2,
                    integration_weights,
                    spectrum_components,
                    parallel_backend,
                    n_jobs_eff,
                    block_size,
                )

        S3_reph = (
            np.zeros((n_w, n_w), dtype=np.complex128)
            if self._wants_rephasing(spectrum_components)
            else None
        )
        S3_unreph = (
            np.zeros((n_w, n_w), dtype=np.complex128)
            if self._wants_unrephasing(spectrum_components)
            else None
        )

        blocks = self._make_w3_blocks(n_w, n_jobs_eff, block_size)

        if verbose:
            print(f"block_size={len(blocks[0]) if blocks else 0}.")

        with self._parallel_context(blas_threads):
            results = self._run_w3_blocks(
                blocks,
                w_list,
                tau2,
                integration_weights,
                parallel_backend,
                n_jobs_eff,
                spectrum_components,
            )

        for i, col_reph, col_unreph in results:
            if S3_reph is not None:
                S3_reph[:, i] = col_reph
            if S3_unreph is not None:
                S3_unreph[:, i] = col_unreph

        return self._format_spectra_result(
            S3_reph, S3_unreph, spectrum_components
        )

    def generate_2D_pathways(
        self,
        w_list,
        tau2,
        pathways=None,
        k_array=None,
        verbose=True,
        k_weights=None,
    ):
        """
        Generate a separate complex 2D spectrum for each requested pathway.

        Parameters
        ----------
        pathways : sequence or None
            Pathway names/objects to calculate. ``None`` uses every configured
            pathway.
        k_array : array_like or None
            Optional momentum coordinates retained for API compatibility.
        k_weights : array_like or None
            Explicit complete integration weights, one per k-point.

        Returns
        -------
        dict
            Mapping ``pathway_name -> complex spectrum``.
        """
        if self.N_k is None:
            raise RuntimeError("Call feed_model() before generate_2D_pathways().")

        selected = (
            list(self.pathways)
            if pathways is None
            else [self._resolve_pathway(item) for item in pathways]
        )
        if not selected:
            raise ValueError("At least one pathway must be selected.")

        w_list = np.asarray(w_list, dtype=float)
        n_w = len(w_list)
        integration_weights = self._resolve_k_weights(k_array, k_weights)

        self._dense_resolvent_cache.clear()
        self._dense_time_cache.clear()
        spectra = {
            pathway.name: np.zeros(
                (n_w, n_w), dtype=np.complex128
            )
            for pathway in selected
        }

        if verbose:
            names = ", ".join(pathway.name for pathway in selected)
            print(
                f"Calculating {len(selected)} pathway spectrum/s on a "
                f"{n_w}x{n_w} grid: {names}."
            )

        for i, w3 in enumerate(w_list):
            for j, w1 in enumerate(w_list):
                for pathway in selected:
                    response = self.calc_pathway(pathway, w3, w1, tau2)
                    spectra[pathway.name][j, i] = self._integrate_k_response(
                        response, integration_weights
                    )
        return spectra

    def benchmark_2D_parallel_backends(
        self,
        w_list,
        tau2,
        k_array=None,
        backends=("serial", "threading", "loky", "process"),
        n_jobs_values=(1, 2, 4),
        block_size=None,
        blas_threads=1,
        repeats=1,
        max_w_points=None,
        k_weights=None,
    ):
        """
        Benchmark omega-loop parallel backends for generate_2D_spectra().

        A short unmeasured run warms up numerical libraries. The first serial
        configuration is then used as the numerical reference. Timings report
        the median of all repetitions.
        """
        n_w = len(w_list)
        d = self.dim
        d2 = d**2 if d is not None else None

        print("\n--- 2D parallel benchmark context ---")
        print(f"Liouville backend      : {self._active_backend}")
        print(f"Hilbert dimension      : {d}")
        print(f"Liouville dimension    : {d2}")
        print(f"k-points               : {self.N_k}")
        print(f"omega points           : {n_w}")
        print(f"omega grid             : {n_w} x {n_w} = {n_w**2} omega pairs")
        print(f"tau2                   : {tau2}")
        print(f"Eta                    : {self.eta}")
        print(f"BLAS threads limit     : {blas_threads}")
        print(f"tested backends        : {backends}")
        print(f"tested n_jobs          : {n_jobs_values}")
        print(f"block_size             : {block_size}")
        print(f"repeats                : {repeats}")

        print("-------------------------------------\n")





        w_bench = np.asarray(w_list, dtype=float)
        if max_w_points is not None:
            w_bench = w_bench[: int(max_w_points)]

        warmup_w = w_bench[: min(8, len(w_bench))]
        if len(warmup_w):
            self.generate_2D_spectra(
                warmup_w,
                tau2,
                k_array=k_array,
                k_weights=k_weights,
                n_jobs=1,
                parallel_backend="serial",
                block_size=block_size,
                blas_threads=blas_threads,
                verbose=False,
            )

        reference = None
        reference_time = None
        rows = []

        for backend in backends:
            jobs_to_run = (1,) if backend == "serial" else n_jobs_values
            for n_jobs in jobs_to_run:
                elapsed_values = []
                spectra = None

                for _ in range(max(1, int(repeats))):
                    t0 = time.perf_counter()
                    spectra = self.generate_2D_spectra(
                        w_bench,
                        tau2,
                        k_array=k_array,
                        k_weights=k_weights,
                        n_jobs=n_jobs,
                        parallel_backend=backend,
                        block_size=block_size,
                        blas_threads=blas_threads,
                        verbose=False,
                    )
                    elapsed = time.perf_counter() - t0
                    elapsed_values.append(elapsed)

                if reference is None:
                    reference = spectra
                    reference_time = float(np.median(elapsed_values))

                max_abs_diff = float(
                    np.max(
                        np.abs(spectra["absorptive"] - reference["absorptive"])
                    )
                )
                median_elapsed = float(np.median(elapsed_values))
                min_elapsed = float(np.min(elapsed_values))
                max_elapsed = float(np.max(elapsed_values))
                speedup = (
                    reference_time / median_elapsed
                    if median_elapsed > 0
                    else np.inf
                )
                row = {
                    "backend": backend,
                    "n_jobs": self._effective_n_jobs(n_jobs),
                    "block_size": block_size,
                    "elapsed_s": median_elapsed,
                    "elapsed_min_s": min_elapsed,
                    "elapsed_max_s": max_elapsed,
                    "speedup_vs_serial": speedup,
                    "max_abs_diff": max_abs_diff,
                }
                rows.append(row)
                print(
                    f"{backend:>15} n_jobs={row['n_jobs']:<3} "
                    f"median={median_elapsed:8.3f}s "
                    f"range=[{min_elapsed:.3f}, {max_elapsed:.3f}]s "
                    f"speedup={speedup:6.2f}x "
                    f"max_abs_diff={max_abs_diff:.3e}"
                )

        return rows




class SpectroscopyPlotter:
    """Plot three pathways and their sum on a shared 2D grid."""

    _COMPONENTS = {
        "real": ("Real", np.real, "bwr", None),
        "imag": ("Imaginary", np.imag, "bwr", None),
        "abs": ("Absolute", np.abs, "magma", 0),
    }
    _COMPONENT_ALIASES = {
        "all": "all", "real": "real", "re": "real",
        "imag": "imag", "imaginary": "imag", "im": "imag",
        "abs": "abs", "absolute": "abs", "magnitude": "abs",
    }

    def __init__(self, w_list=None, detection_phase=np.pi / 2):
        self.w_list = (
            None if w_list is None else np.asarray(w_list, dtype=float)
        )
        self.detection_phase = 0.0 if detection_phase is None else float(detection_phase)

    def _apply_detection_phase(self, data):
        return np.exp(1j * self.detection_phase) * np.asarray(data)

    @staticmethod
    def _plot_subplot(ax, w, data, levels, cmap, title, xlabel, ylabel, vmin):
        limit = np.max(np.abs(data))
        if not np.isfinite(limit):
            raise ValueError("Spectrum contains non-finite values.")
        if limit == 0:
            limit = np.finfo(float).eps
        lower = -limit if vmin is None else vmin
        contour = ax.contourf(w, w, data, levels, cmap=cmap, vmin=lower, vmax=limit)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.figure.colorbar(contour, ax=ax)

    def plot_1d(
        self,
        signal,
        w=None,
        component="real",
        title="1D spectrum",
        xlabel=r"$\omega$",
        ylabel=None,
        label=None,
        reference_positions=None,
        peak_positions=None,
        normalize=False,
        save_path=None,
        show=True,
        ax=None,
    ):
        """Plot one complex one-dimensional spectrum.

        Parameters
        ----------
        signal : array_like
            Complex spectrum sampled on ``w``.
        w : array_like or None
            Frequency axis. Defaults to the axis supplied to the plotter.
        component : {"real", "imag", "abs"}
            Quadrature displayed after applying ``detection_phase``.
        reference_positions : sequence or None
            Expected resonance positions, drawn as dashed vertical lines.
        peak_positions : sequence or None
            Extracted resonance positions, drawn on the spectrum.
        normalize : bool
            Divide the displayed quadrature by its maximum absolute value.
        """
        w = self.w_list if w is None else np.asarray(w, dtype=float)
        signal = np.asarray(signal)
        if w.ndim != 1 or w.size == 0:
            raise ValueError("w must be a non-empty one-dimensional array.")
        if signal.ndim != 1 or signal.shape != w.shape:
            raise ValueError(
                f"signal has shape {signal.shape}; expected {w.shape}."
            )

        component_key = self._COMPONENT_ALIASES.get(str(component).lower())
        if component_key not in self._COMPONENTS:
            raise ValueError("component must be 'real', 'imag', or 'abs'.")
        component_label, transform, _, _ = self._COMPONENTS[component_key]
        data = transform(self._apply_detection_phase(signal))
        if not np.all(np.isfinite(data)):
            raise ValueError("Spectrum contains non-finite values.")
        if normalize:
            scale = np.max(np.abs(data))
            if scale > 0:
                data = data / scale

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.5))
        else:
            fig = ax.figure
        ax.plot(w, data, color="black", label=label)

        if reference_positions is not None:
            for index, position in enumerate(reference_positions):
                ax.axvline(
                    position,
                    color="tab:red",
                    linestyle="--",
                    alpha=0.8,
                    label="Reference positions" if index == 0 else None,
                )
        if peak_positions is not None:
            peak_positions = np.asarray(peak_positions, dtype=float)
            ax.scatter(
                peak_positions,
                np.interp(peak_positions, w, data),
                color="tab:blue",
                zorder=3,
                label="Extracted positions",
            )

        if ylabel is None:
            ylabel = "Normalized signal" if normalize else f"{component_label} signal"
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.2)
        if label is not None or reference_positions is not None or peak_positions is not None:
            ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax

    @staticmethod
    def _selected_pathway_names(result, pathways):
        available = tuple(result.pathways)
        if pathways is None or pathways == "all":
            return available
        if isinstance(pathways, str):
            pathways = (pathways,)
        selected = tuple(str(name) for name in pathways)
        unknown = [name for name in selected if name not in result.pathways]
        if unknown:
            raise KeyError(
                f"Unknown pathway name(s) {unknown}; available={available}."
            )
        if len(set(selected)) != len(selected):
            raise ValueError("pathways contains duplicate names.")
        if not selected:
            raise ValueError("At least one pathway must be selected.")
        return selected

    @staticmethod
    def _selected_totals(result, selected_names, totals):
        if totals is None or totals is False:
            return {}
        if totals == "selected":
            return {
                "Selected total": sum(
                    np.asarray(result.pathways[name]) for name in selected_names
                )
            }
        if totals == "auto":
            if set(selected_names) != set(result.pathways):
                return {
                    "Selected total": sum(
                        np.asarray(result.pathways[name])
                        for name in selected_names
                    )
                }
            nq_totals = [
                name for name in result.components
                if re.fullmatch(r"\d+Q", str(name))
            ]
            if nq_totals:
                totals = nq_totals
            else:
                components = []
                for name in selected_names:
                    metadata = result.pathway_metadata.get(name, {})
                    component = metadata.get("component")
                    if component in result.components and component not in components:
                        components.append(component)
                totals = components or list(result.components)
        elif isinstance(totals, str):
            totals = (totals,)

        selected_totals = {}
        for name in totals:
            if name not in result.components:
                raise KeyError(
                    f"Unknown component total {name!r}; "
                    f"available={tuple(result.components)}."
                )
            selected_totals[f"Total {name}"] = np.asarray(
                result.components[name]
            )
        return selected_totals

    @staticmethod
    def _render_ufss_pathway_diagrams(
        result,
        pathway_names,
        *,
        diagram_size,
        display_diagrams,
        save_pdf,
        output_directory,
    ):
        try:
            import ufss
        except ImportError as exc:
            raise ImportError(
                "UFSS is required to render pathway diagrams."
            ) from exc

        generator = ufss.DiagramGenerator(detection_type="polarization")
        generator.diagram_size = diagram_size
        generator.include_state_labels = True
        generator.include_pulse_labels = True
        generator.include_emission_arrow = True

        diagrams = {}
        diagram_paths = {}
        diagram_directory = None
        if save_pdf:
            diagram_directory = Path(output_directory) / "Feynman_diagrams"
            diagram_directory.mkdir(parents=True, exist_ok=True)

        for name in pathway_names:
            metadata = result.pathway_metadata.get(name)
            if metadata is None or not metadata.get("ufss_diagram"):
                raise ValueError(
                    f"Pathway {name!r} has no retained UFSS diagram metadata."
                )
            diagram = tuple(
                (str(interaction), int(pulse_index))
                for interaction, pulse_index in metadata["ufss_diagram"]
            )
            pulse_count = max((item[1] for item in diagram), default=-1) + 1
            generator.pulse_labels = [
                str(index) for index in range(1, pulse_count + 1)
            ]
            generator.draw_diagram(diagram)
            canvas = generator.c
            diagrams[name] = canvas

            if display_diagrams:
                try:
                    from IPython.display import display
                except ImportError as exc:
                    raise ImportError(
                        "IPython is required when display_diagrams=True."
                    ) from exc
                display(canvas, exclude="image/png")

            if save_pdf:
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
                pdf_path = diagram_directory / f"{safe_name}.pdf"
                canvas.writePDFfile(str(pdf_path.with_suffix("")))
                diagram_paths[name] = pdf_path

        return diagrams, diagram_paths

    def plot_pathways_multiorder(
        self,
        result,
        pathways="all",
        totals="auto",
        view="real",
        normalization="shared",
        ncols=None,
        levels=30,
        axis_labels=None,
        include_diagrams=True,
        diagram_size="medium",
        display_diagrams=False,
        save_pdf=False,
        output_directory=None,
        spectrum_pdf_name="pathway_spectra.pdf",
        show=True,
    ):
        """Plot arbitrary-order pathway spectra and matching UFSS diagrams.

        Individual spectra and component totals come from ``SpectrumResult``.
        Retained UFSS instructions are rendered with the same pathway names.
        PDF output is opt-in; when ``save_pdf`` is false no directory or file
        is created by this method.
        """
        if not isinstance(result, SpectrumResult):
            raise TypeError("result must be a SpectrumResult.")
        if len(result.axis_names) != 2 or len(result.axis_values) != 2:
            raise ValueError("A two-frequency SpectrumResult is required.")
        if save_pdf and output_directory is None:
            raise ValueError(
                "output_directory is required when save_pdf=True."
            )

        selected_names = self._selected_pathway_names(result, pathways)
        selected_totals = self._selected_totals(
            result, selected_names, totals
        )
        panel_data = {
            **{name: np.asarray(result.pathways[name]) for name in selected_names},
            **selected_totals,
        }

        y_values = np.asarray(result.axis_values[0], dtype=float)
        x_values = np.asarray(result.axis_values[1], dtype=float)
        expected_shape = (y_values.size, x_values.size)
        for name, values in panel_data.items():
            if values.shape != expected_shape:
                raise ValueError(
                    f"Panel {name!r} has shape {values.shape}; "
                    f"expected {expected_shape}."
                )

        view_key = self._COMPONENT_ALIASES.get(str(view).lower())
        if view_key not in self._COMPONENTS:
            raise ValueError("view must be 'real', 'imag', or 'abs'.")
        view_label, transform, cmap, absolute_vmin = self._COMPONENTS[view_key]
        phased = {
            name: transform(self._apply_detection_phase(values))
            for name, values in panel_data.items()
        }
        if not all(np.all(np.isfinite(values)) for values in phased.values()):
            raise ValueError("A pathway spectrum contains non-finite values.")

        normalization = str(normalization).lower()
        if normalization not in {"shared", "individual", "none"}:
            raise ValueError(
                "normalization must be 'shared', 'individual', or 'none'."
            )
        global_scale = max(
            (float(np.max(np.abs(values))) for values in phased.values()),
            default=0.0,
        )
        if global_scale == 0:
            global_scale = np.finfo(float).eps
        if normalization == "shared":
            display_data = {
                name: values / global_scale for name, values in phased.items()
            }
            shared_limit = 1.0
            colorbar_label = f"Normalized {view_label.lower()} signal"
        else:
            display_data = dict(phased)
            shared_limit = global_scale if normalization == "none" else None
            colorbar_label = f"{view_label} signal"
            if normalization == "individual":
                for name, values in display_data.items():
                    scale = float(np.max(np.abs(values)))
                    if scale > 0:
                        display_data[name] = values / scale
                colorbar_label = f"Individually normalized {view_label.lower()} signal"

        panel_names = tuple(display_data)
        panel_count = len(panel_names)
        levels = int(levels)
        if levels < 2:
            raise ValueError("levels must be at least two.")
        if ncols is None:
            ncols = int(math.ceil(math.sqrt(panel_count)))
        ncols = int(ncols)
        if ncols < 1:
            raise ValueError("ncols must be positive.")
        nrows = int(math.ceil(panel_count / ncols))
        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.2 * ncols, 4.4 * nrows),
            squeeze=False,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axis_labels = {} if axis_labels is None else dict(axis_labels)
        x_label = axis_labels.get(result.axis_names[1], result.axis_names[1])
        y_label = axis_labels.get(result.axis_names[0], result.axis_names[0])

        for axis, name in zip(axes.flat, panel_names):
            values = display_data[name]
            limit = shared_limit
            if limit is None:
                limit = float(np.max(np.abs(values)))
                if limit == 0:
                    limit = np.finfo(float).eps
            vmin = 0.0 if absolute_vmin == 0 else -limit
            contour_levels = np.linspace(vmin, limit, levels)
            contour = axis.contourf(
                x_values,
                y_values,
                values,
                levels=contour_levels,
                cmap=cmap,
                vmin=vmin,
                vmax=limit,
            )
            metadata = result.pathway_metadata.get(name, {})
            if metadata:
                interactions = " ".join(metadata.get("interactions", ()))
                coherences = metadata.get("coherence_orders", ())
                title = f"{name}: {interactions}\nq={coherences}"
            else:
                title = name
            axis.set(title=title, xlabel=x_label, ylabel=y_label)
            figure.colorbar(contour, ax=axis, label=colorbar_label)
        for axis in axes.flat[panel_count:]:
            axis.set_visible(False)

        spectrum_pdf = None
        if save_pdf:
            output_directory = Path(output_directory)
            output_directory.mkdir(parents=True, exist_ok=True)
            spectrum_pdf = output_directory / spectrum_pdf_name
            if spectrum_pdf.suffix.lower() != ".pdf":
                spectrum_pdf = spectrum_pdf.with_suffix(".pdf")
            figure.savefig(spectrum_pdf, bbox_inches="tight")

        diagrams = {}
        diagram_paths = {}
        if include_diagrams:
            diagrams, diagram_paths = self._render_ufss_pathway_diagrams(
                result,
                selected_names,
                diagram_size=diagram_size,
                display_diagrams=display_diagrams,
                save_pdf=save_pdf,
                output_directory=output_directory,
            )

        if show:
            plt.show()
        return PathwayPlotResult(
            figure=figure,
            axes=axes,
            panel_names=panel_names,
            diagrams=diagrams,
            diagram_paths=diagram_paths,
            spectrum_pdf=spectrum_pdf,
        )

    def plot_pathways_grid(
        self,
        pathways_dict,
        signal_type="rephasing",
        total_signal=None,
        w=None,
        levels=20,
        save_path=None,
        show=True,
        zoom_quadrant=True,
        zoom_bounds=None,
        component="all",
    ):
        """Plot R1-R3 or R4-R6 and their sum for selected components."""
        w = self.w_list if w is None else np.asarray(w, dtype=float)
        if w.ndim != 1 or w.size == 0:
            raise ValueError("w must be a non-empty one-dimensional array.")

        signal_key = str(signal_type).lower().replace("-", "")
        signal_definitions = {
            "rephasing": (("R1", "R2", "R3"), -1, "Rephasing"),
            "unrephasing": (("R4", "R5", "R6"), 1, "Non-rephasing"),
            "nonrephasing": (("R4", "R5", "R6"), 1, "Non-rephasing"),
        }
        if signal_key not in signal_definitions:
            raise ValueError("signal_type must be 'rephasing' or 'unrephasing'.")
        pathway_names, y_sign, display_name = signal_definitions[signal_key]

        pathway_data = []
        for name in pathway_names:
            value = pathways_dict.get(name, pathways_dict.get(int(name[1:])))
            if value is None:
                raise KeyError(f"Missing pathway {name!r}.")
            value = np.asarray(value)
            if value.shape != (w.size, w.size):
                raise ValueError(
                    f"Pathway {name!r} has shape {value.shape}; expected {(w.size, w.size)}."
                )
            pathway_data.append(value)

        if total_signal is None:
            total_signal = sum(pathway_data)
        total_signal = np.asarray(total_signal)
        if total_signal.shape != (w.size, w.size):
            raise ValueError(
                f"total_signal has shape {total_signal.shape}; expected {(w.size, w.size)}."
            )

        component_key = self._COMPONENT_ALIASES.get(str(component).lower())
        if component_key is None:
            raise ValueError("component must be 'all', 'real', 'imag', or 'abs'.")
        component_keys = tuple(self._COMPONENTS) if component_key == "all" else (component_key,)

        if zoom_bounds is not None:
            if len(zoom_bounds) != 4:
                raise ValueError("zoom_bounds must contain (x_min, x_max, y_min, y_max).")
            x_min, x_max, y_min, y_max = map(float, zoom_bounds)
            bounds = (x_min, x_max, y_min, y_max)
            if not np.all(np.isfinite(bounds)) or x_min >= x_max or y_min >= y_max:
                raise ValueError("zoom_bounds must be finite and strictly increasing.")

        phased_data = [self._apply_detection_phase(data) for data in (*pathway_data, total_signal)]
        column_titles = (*pathway_names, f"Total {display_name}")
        fig, axes = plt.subplots(
            len(component_keys), 4, figsize=(20, 4.7 * len(component_keys)),
            sharex=True, sharey=True, squeeze=False,
        )

        for row, key in enumerate(component_keys):
            label, transform, cmap, vmin = self._COMPONENTS[key]
            for column, data in enumerate(phased_data):
                ax = axes[row, column]
                self._plot_subplot(
                    ax, w, transform(data), levels, cmap,
                    f"{label} / {column_titles[column]}",
                    r"$\omega_3$" if row == len(component_keys) - 1 else "",
                    r"$\omega_1$" if column == 0 else "", vmin,
                )
                diagonal_extent = min(float(np.max(w)), abs(float(np.min(w))))
                ax.plot(
                    [0, diagonal_extent], [0, y_sign * diagonal_extent],
                    color="white", linestyle="--", linewidth=1.0, alpha=0.7,
                )
                if zoom_bounds is not None:
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                elif zoom_quadrant:
                    ax.set_xlim(0, np.max(w))
                    ax.set_ylim(np.min(w), 0) if y_sign < 0 else ax.set_ylim(0, np.max(w))

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig, axes
