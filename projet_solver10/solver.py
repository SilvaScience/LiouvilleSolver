"""Public orchestration layer for SolverV10.

This module connects sector-based physical models to numerical backends.
It selects a compatible backend, validates model requirements, and manages
the active spectroscopy pathways. The solver can evaluate one response or
build multidimensional grids while reusing each pathway propagation for
additional observables. It also exposes compact summaries, memory estimates,
and direct access to supported Hilbert-space resolvents. Physical definitions
remain in the model; numerical details remain in the backend.
"""

from collections.abc import Mapping

import numpy as np

from .backends import (
    DenseLiouvilleBackend,
    SparseSectorBackend,
)
from .capabilities import Capabilities, ModelRequirements
from .exceptions import CapabilityError
from .observables import ObservableSpec, normalize_observables
from .pathways import FrequencyPathway
from .protocols import SpectroscopyProtocol
from .results import SpectrumResult


class SpectroscopySolver:
    """Coordinate an external model with a numerical backend."""

    _BACKEND_CLASSES = {
        "dense": DenseLiouvilleBackend,
        "dense_liouville": DenseLiouvilleBackend,
        "sparse": SparseSectorBackend,
        "sparse_sector": SparseSectorBackend,
    }

    def __init__(
        self,
        *,
        backend="sparse_sector",
        eta=0.05,
        requirements=None,
        thermodynamic_context=None,
        **backend_options,
    ):
        self.backend_name = str(backend).lower()
        self.eta = float(eta)
        self.backend_options = dict(backend_options)
        self.requirements = (
            None
            if requirements is None
            else ModelRequirements.from_value(requirements)
        )
        self.thermodynamic_context = thermodynamic_context
        self.model_requirements = None
        self.model = None
        self.backend = None
        self.pathways = []

    @classmethod
    def register_backend(cls, name, backend_class):
        """Backend name and class -> registered backend."""
        name = str(name).lower()
        if not name:
            raise ValueError("Backend name cannot be empty.")
        if not callable(backend_class):
            raise TypeError("backend_class must be callable.")
        cls._BACKEND_CLASSES[name] = backend_class

    @classmethod
    def available_backends(cls):
        """Return the registered backend names."""
        return tuple(sorted(cls._BACKEND_CLASSES))

    def _resolve_model_requirements(self, model):
        if self.requirements is not None:
            return self.requirements
        method = getattr(model, "requirements", None)
        if callable(method):
            return ModelRequirements.from_value(method())

        capabilities = getattr(model, "capabilities", None)
        capabilities = capabilities() if callable(capabilities) else capabilities
        legacy = Capabilities.from_value(capabilities)
        domains = []
        if legacy.time_domain:
            domains.append("time")
        if legacy.frequency_domain:
            domains.append("frequency")
        return ModelRequirements(
            state_kind=(
                "mixed"
                if legacy.mixed_state
                or self.thermodynamic_context is not None
                else "pure"
            ),
            generator_kind="lindblad" if legacy.lindblad else "unitary",
            domains=tuple(domains or ("time",)),
            matrix_free_required=False,
            metadata={"legacy_capabilities": True},
        )

    def _resolve_backend_class(self, model):
        requirements = self._resolve_model_requirements(model)
        self.model_requirements = requirements
        if self.backend_name != "auto":
            try:
                backend_class = self._BACKEND_CLASSES[self.backend_name]
            except KeyError as exc:
                supported = ", ".join(self.available_backends())
                raise ValueError(
                    f"Unknown backend {self.backend_name!r}. "
                    f"Available values: {supported}."
                ) from exc
            missing = backend_class.backend_capabilities.missing_for(
                requirements
            )
            if missing:
                raise CapabilityError(
                    f"Backend {backend_class.name!r} does not satisfy: "
                    + ", ".join(missing)
                )
            return backend_class

        candidates = (
            SparseSectorBackend,
            DenseLiouvilleBackend,
        )
        supported = [
            backend_class
            for backend_class in candidates
            if backend_class.backend_capabilities.supports(requirements)
        ]
        if not supported:
            raise CapabilityError(
                "No backend satisfies the model requirements: "
                f"{requirements.as_dict()}."
            )
        return supported[0]

    def feed_model(self, model, *, context=None):
        """Sector-based model -> initialized backend and solver."""
        if context is not None:
            self.thermodynamic_context = context
        backend_class = self._resolve_backend_class(model)
        backend = backend_class(
            eta=self.eta,
            **self.backend_options,
        )
        backend.build(model, context=self.thermodynamic_context)
        actual_requirements = ModelRequirements(
            state_kind=(
                "pure" if backend._initial_is_pure else "mixed"
            ),
            generator_kind=backend.generator.kind,
            domains=self.model_requirements.domains,
            exactness=self.model_requirements.exactness,
            matrix_free_required=(
                self.model_requirements.matrix_free_required
            ),
            stationary_mode_deflation_required=(
                self.model_requirements.stationary_mode_deflation_required
            ),
            metadata=dict(self.model_requirements.metadata),
        )
        missing = backend.backend_capabilities.missing_for(
            actual_requirements
        )
        if missing:
            raise CapabilityError(
                f"After initialization, backend {backend.name!r} does not "
                "satisfy the effective model requirements: "
                + ", ".join(missing)
            )
        self.model_requirements = actual_requirements
        self.model = model
        self.backend = backend
        return self

    def set_pathways(self, pathways):
        """Pathway definitions -> active pathway list."""
        normalized = []
        names = set()
        for item in pathways:
            pathway = (
                item
                if isinstance(item, FrequencyPathway)
                else FrequencyPathway(**item)
            )
            if pathway.name in names:
                raise ValueError(f"Duplicate pathway name: {pathway.name!r}")
            names.add(pathway.name)
            normalized.append(pathway)
        if not normalized:
            raise ValueError("At least one pathway is required.")
        self.pathways = normalized
        return self

    def get_pathways(self, component=None):
        """Optional component -> matching pathways."""
        if component is None:
            return list(self.pathways)
        component = str(component).lower().replace("-", "")
        if component in {"nonrephasing", "nonrephase"}:
            component = "unrephasing"
        return [
            pathway
            for pathway in self.pathways
            if pathway.component == component
        ]

    def pathway_summary(self):
        """Return serializable metadata for the active pathways."""
        return [pathway.metadata() for pathway in self.pathways]

    def _require_ready(self):
        if self.backend is None:
            raise RuntimeError("Call feed_model(model) before calculation.")

    def _resolve_pathway(self, pathway):
        if isinstance(pathway, FrequencyPathway):
            return pathway
        name = str(pathway)
        matches = [item for item in self.pathways if item.name == name]
        if not matches:
            raise KeyError(f"Unknown pathway: {name!r}")
        return matches[0]

    def calc_pathway(self, pathway, protocol, coordinates):
        """Pathway, protocol, and coordinates -> pathway response."""
        self._require_ready()
        pathway = self._resolve_pathway(pathway)
        if not isinstance(protocol, SpectroscopyProtocol):
            raise TypeError("protocol must be a SpectroscopyProtocol.")
        return self.backend.calc_pathway(
            pathway,
            protocol,
            dict(coordinates),
        )

    def jump_channel_names(self):
        """Return GKSL channels available as jump observables."""
        self._require_ready()
        return self.backend.jump_channel_names()

    def calc_pathway_observables(
        self, pathway, protocol, coordinates, observables=None
    ):
        """Pathway propagation -> responses for multiple observables."""
        self._require_ready()
        pathway = self._resolve_pathway(pathway)
        if not isinstance(protocol, SpectroscopyProtocol):
            raise TypeError("protocol must be a SpectroscopyProtocol.")
        defaults = (ObservableSpec(name=pathway.detection),)
        requested = normalize_observables(observables) if observables is not None else ()
        specs = normalize_observables(
            None,
            defaults=defaults + tuple(requested),
        )
        return self.backend.calc_pathway_observables(
            pathway,
            protocol,
            dict(coordinates),
            specs,
        )

    def calc_component(self, component, protocol, coordinates, pathways=None):
        """Component pathways -> summed response."""
        candidates = (
            self.get_pathways(component)
            if pathways is None
            else [self._resolve_pathway(item) for item in pathways]
        )
        normalized = str(component).lower().replace("-", "")
        if normalized in {"nonrephasing", "nonrephase"}:
            normalized = "unrephasing"
        selected = [
            item for item in candidates if item.component == normalized
        ]
        if not selected:
            raise ValueError(
                f"No pathway found for component {component!r}."
            )
        value = 0.0j
        diagnostics = {}
        for pathway in selected:
            result = self.calc_pathway(pathway, protocol, coordinates)
            value += result.value
            diagnostics[pathway.name] = result.diagnostics
        return complex(value), diagnostics

    def _normalize_axes(self, protocol, axes, fixed_coordinates):
        if not isinstance(axes, Mapping) or not axes:
            raise TypeError("axes must be a non-empty mapping {name: values}.")
        normalized_axes = {}
        for name, values in axes.items():
            name = str(name)
            if name not in protocol.coordinate_names:
                raise ValueError(
                    f"Axis {name!r} is not part of protocol "
                    f"{protocol.name!r}."
                )
            values = np.asarray(values)
            if values.ndim != 1 or values.size == 0:
                raise ValueError(
                    f"Axis {name!r} must be a non-empty 1D array."
                )
            normalized_axes[name] = values

        overlap = sorted(set(normalized_axes).intersection(fixed_coordinates))
        if overlap:
            raise ValueError(
                f"Coordinates supplied as both axes and fixed values: {overlap}"
            )
        supplied = set(normalized_axes).union(fixed_coordinates)
        missing = sorted(set(protocol.coordinate_names).difference(supplied))
        if missing:
            raise KeyError(
                f"Undefined protocol coordinates: {missing}"
            )
        return normalized_axes

    def generate_spectrum(
        self,
        protocol,
        axes,
        *,
        fixed_coordinates=None,
        pathways=None,
        observables=None,
    ):
        """Protocol and coordinate grid -> multidimensional responses.

        Additional observables reuse each pathway propagation.
        """
        self._require_ready()
        if not isinstance(protocol, SpectroscopyProtocol):
            raise TypeError("protocol must be a SpectroscopyProtocol.")
        fixed_coordinates = dict(fixed_coordinates or {})
        normalized_axes = self._normalize_axes(
            protocol, axes, fixed_coordinates
        )
        selected = (
            list(self.pathways)
            if pathways is None
            else [self._resolve_pathway(item) for item in pathways]
        )
        if not selected:
            raise ValueError("No pathway selected.")

        defaults = tuple(
            ObservableSpec(name=pathway.detection) for pathway in selected
        )
        requested = (
            ()
            if observables is None
            else normalize_observables(observables)
        )
        observable_specs = normalize_observables(
            None,
            defaults=defaults + tuple(requested),
        )

        axis_names = tuple(normalized_axes)
        axis_values = tuple(normalized_axes[name] for name in axis_names)
        shape = tuple(values.size for values in axis_values)
        pathway_values = {
            pathway.name: np.zeros(shape, dtype=np.complex128)
            for pathway in selected
        }
        diagnostics = {pathway.name: {} for pathway in selected}
        observable_values = {
            spec.name: {
                pathway.name: np.zeros(shape, dtype=np.complex128)
                for pathway in selected
            }
            for spec in observable_specs
        }
        self.backend.clear_caches()

        for grid_index in np.ndindex(shape):
            coordinates = dict(fixed_coordinates)
            coordinates.update(
                {
                    name: values[index]
                    for name, values, index in zip(
                        axis_names, axis_values, grid_index
                    )
                }
            )
            for pathway in selected:
                result = self.backend.calc_pathway_observables(
                    pathway,
                    protocol,
                    coordinates,
                    observable_specs,
                )
                pathway_values[pathway.name][grid_index] = result.value
                for name, value in result.observables.items():
                    observable_values[name][pathway.name][grid_index] = value
                if result.diagnostics:
                    diagnostics[pathway.name][grid_index] = result.diagnostics

        components = {}
        for pathway in selected:
            if pathway.component not in components:
                components[pathway.component] = np.zeros(
                    shape, dtype=np.complex128
                )
            components[pathway.component] += pathway_values[pathway.name]

        observable_components = {}
        for spec in observable_specs:
            component_values = {}
            for pathway in selected:
                if pathway.component not in component_values:
                    component_values[pathway.component] = np.zeros(
                        shape, dtype=np.complex128
                    )
                component_values[pathway.component] += observable_values[
                    spec.name
                ][pathway.name]
            observable_components[spec.name] = component_values

        return SpectrumResult(
            axis_names=axis_names,
            axis_values=axis_values,
            pathways=pathway_values,
            components=components,
            fixed_coordinates=fixed_coordinates,
            pathway_metadata={
                pathway.name: pathway.metadata() for pathway in selected
            },
            diagnostics={
                "backend": self.backend.name,
                "pathways": diagnostics,
                "observables": tuple(spec.name for spec in observable_specs),
            },
            observables=observable_values,
            observable_components=observable_components,
        )

    def generate_nq_spectrum(
        self,
        order,
        protocol,
        axes,
        *,
        fixed_coordinates=None,
        pathways=None,
        observables=None,
    ):
        """NQ protocol and grid -> separate ``+NQ`` and ``-NQ`` responses."""
        order = abs(int(order))
        matching_intervals = [
            index
            for index, interval in enumerate(protocol.intervals)
            if interval.coherence_order is not None
            and abs(interval.coherence_order) == order
        ]
        if len(matching_intervals) != 1:
            raise ValueError(
                "Protocol must identify exactly one "
                f"{order}Q coherence interval."
            )
        coherence_index = matching_intervals[0]
        candidates = (
            list(self.pathways)
            if pathways is None
            else [self._resolve_pathway(item) for item in pathways]
        )
        selected = [
            pathway
            for pathway in candidates
            if abs(pathway.coherence_orders[coherence_index]) == order
        ]
        if not selected:
            raise ValueError(f"No pathway carries {order}Q coherence.")

        result = self.generate_spectrum(
            protocol,
            axes,
            fixed_coordinates=fixed_coordinates,
            pathways=selected,
            observables=observables,
        )
        shape = result.shape
        components = dict(result.components)
        if order == 0:
            total = np.zeros(shape, dtype=np.complex128)
            for pathway in selected:
                total += result.pathways[pathway.name]
            components["0Q"] = total
        else:
            positive = np.zeros(shape, dtype=np.complex128)
            negative = np.zeros(shape, dtype=np.complex128)
            for pathway in selected:
                target = (
                    positive
                    if pathway.coherence_orders[coherence_index] > 0
                    else negative
                )
                target += result.pathways[pathway.name]
            components[f"+{order}Q"] = positive
            components[f"-{order}Q"] = negative
            components[f"{order}Q"] = positive + negative
        result.components = components

        observable_components = dict(result.observable_components)
        for name, pathway_values in result.observables.items():
            if order == 0:
                total = np.zeros(shape, dtype=np.complex128)
                for pathway in selected:
                    total += pathway_values[pathway.name]
                component_values = {"0Q": total}
            else:
                positive = np.zeros(shape, dtype=np.complex128)
                negative = np.zeros(shape, dtype=np.complex128)
                for pathway in selected:
                    target = (
                        positive
                        if pathway.coherence_orders[coherence_index] > 0
                        else negative
                    )
                    target += pathway_values[pathway.name]
                component_values = {
                    f"+{order}Q": positive,
                    f"-{order}Q": negative,
                    f"{order}Q": positive + negative,
                }
            observable_components[name] = component_values
        result.observable_components = observable_components
        return result

    def estimate_memory(self):
        """Backend layout -> minimum structural memory estimates."""
        self._require_ready()
        dimension = self.backend.layout.total_dimension
        complex_bytes = np.dtype(np.complex128).itemsize
        result = {
            "hilbert_dimension": dimension,
            "one_state_vector_bytes": dimension * complex_bytes,
            "backend": self.backend.name,
        }
        if self.backend.name == "dense_liouville":
            liouville_dimension = dimension * dimension
            result.update(
                {
                    "liouville_dimension": liouville_dimension,
                    "one_density_vector_bytes": (
                        liouville_dimension * complex_bytes
                    ),
                    "one_liouville_matrix_bytes": (
                        liouville_dimension
                        * liouville_dimension
                        * complex_bytes
                    ),
                }
            )
        return result

    def summary(self):
        """Return the active configuration without exposing model internals."""
        self._require_ready()
        summary = {
            "backend": self.backend.name,
            "backend_capabilities": self.backend.capabilities.as_dict(),
            "backend_support": (
                self.backend.backend_capabilities.as_dict()
            ),
            "model_requirements": self.model_requirements.as_dict(),
            "model_capabilities": (
                self.backend.model_capabilities.as_dict()
            ),
            "sectors": self.backend.layout.sectors,
            "sector_dimensions": dict(self.backend.layout.dimensions),
            "total_dimension": self.backend.layout.total_dimension,
            "pathways": self.pathway_summary(),
        }
        decomposition = getattr(self.model, "transition_decomposition", None)
        if callable(decomposition):
            decomposition = decomposition()
        if decomposition is not None:
            summary["transition_decomposition"] = str(decomposition)
        return summary

    def solve_hilbert_resolvent(self, vector, omega, **kwargs):
        """Vector and frequency -> active backend's Hilbert resolvent."""
        self._require_ready()
        method = getattr(self.backend, "solve_hilbert_resolvent", None)
        if method is None:
            raise CapabilityError(
                f"Backend {self.backend.name!r} does not expose a "
                "Hilbert resolvent."
            )
        return method(vector, omega, **kwargs)
