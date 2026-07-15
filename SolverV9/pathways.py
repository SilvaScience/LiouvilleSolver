"""Pathway construction and UFSS translation helpers."""

import warnings

import numpy as np

from .models import (
    FrequencyPathway,
    _coherence_orders_from_interactions,
    _response_prefactor_from_interactions,
)


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


def phase_discrimination_component(phase_discrimination):
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


def make_pathway_names_unique(pathways):
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
            translated_component = phase_discrimination_component(
                phase_discrimination
            )
        pathways = translate_ufss_diagrams(
            diagrams,
            component=translated_component,
            names=names,
            amplitudes=amplitudes,
            prefactors=prefactors,
            coherence_orders=coherence_orders,
            allow_noncanonical_order=allow_noncanonical_order,
        )
        return pathways

normalize_phase_discrimination = _normalize_phase_discrimination
