"""Liouville pathways and optional sector constraints.

This module represents ordered ket and bra interactions, derives signed
coherence histories, and computes perturbative response prefactors. It also
translates UFSS-style diagram lists into SolverV10 pathways without requiring
UFSS as a runtime dependency.
"""

from dataclasses import dataclass


_COHERENCE_CHANGE = {"Ku": 1, "Kd": -1, "Bu": -1, "Bd": 1}
_KET_LABELS = {"Ku", "Kd"}
_BRA_LABELS = {"Bu", "Bd"}


@dataclass(frozen=True)
class Interaction:
    """Interaction label and optional sectors -> constrained transition."""

    label: str
    operator: str = "light_matter"
    pulse_index: int | None = None
    source_sector: object = None
    target_sector: object = None

    def __post_init__(self):
        label = str(self.label)
        if label not in _COHERENCE_CHANGE:
            raise ValueError(
                f"Unknown interaction {label!r}; use Ku, Kd, Bu, or Bd."
            )
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "operator", str(self.operator))
        if self.pulse_index is not None:
            object.__setattr__(self, "pulse_index", int(self.pulse_index))

    @property
    def side(self):
        return "ket" if self.label in _KET_LABELS else "bra"

    @property
    def direction(self):
        """Interaction label -> branch-vector direction.

        Bra-side right multiplication acts through the adjoint.
        """
        return "plus" if self.label in {"Ku", "Bu"} else "minus"

    @property
    def perturbative_prefactor(self):
        """Return this interaction's pathway-prefactor contribution."""
        return 1j * (-1 if self.side == "bra" else 1)


def coherence_orders_from_interactions(interactions):
    """Ordered interactions -> signed coherence history."""
    current = 0
    history = []
    for interaction in interactions:
        label = (
            interaction.label
            if isinstance(interaction, Interaction)
            else str(interaction)
        )
        current += _COHERENCE_CHANGE[label]
        history.append(current)
    return tuple(history)


@dataclass(frozen=True)
class FrequencyPathway:
    """Generic definition of an ordered pathway."""

    name: str
    interactions: tuple
    component: str = "custom"
    amplitude: complex = 1.0
    prefactor: complex | None = None
    coherence_orders: tuple = ()
    detection: str = "polarization"

    def __post_init__(self):
        interactions = tuple(
            item
            if isinstance(item, Interaction)
            else Interaction(**item)
            if isinstance(item, dict)
            else Interaction(str(item))
            for item in self.interactions
        )
        if not interactions:
            raise ValueError("A pathway must contain interactions.")
        expected = coherence_orders_from_interactions(interactions)
        coherence_orders = (
            tuple(int(item) for item in self.coherence_orders)
            if self.coherence_orders
            else expected
        )
        if coherence_orders != expected:
            raise ValueError(
                f"Pathway {self.name!r} declares {coherence_orders}, "
                f"but its interactions imply {expected}."
            )
        component = str(self.component).lower().replace("-", "")
        if component in {"nonrephasing", "nonrephase"}:
            component = "unrephasing"
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "interactions", interactions)
        object.__setattr__(self, "coherence_orders", coherence_orders)
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "amplitude", complex(self.amplitude))
        object.__setattr__(self, "detection", str(self.detection))
        if self.prefactor is not None:
            object.__setattr__(self, "prefactor", complex(self.prefactor))

    @property
    def bra_sign(self):
        count = sum(item.side == "bra" for item in self.interactions)
        return -1 if count % 2 else 1

    @property
    def response_prefactor(self):
        if self.prefactor is not None:
            return self.prefactor
        return (
            complex(self.amplitude)
            * (1j ** len(self.interactions))
            * self.bra_sign
        )

    def metadata(self):
        return {
            "name": self.name,
            "component": self.component,
            "amplitude": self.amplitude,
            "prefactor": self.prefactor,
            "response_prefactor": self.response_prefactor,
            "coherence_orders": self.coherence_orders,
            "detection": self.detection,
            "interactions": tuple(
                {
                    "label": item.label,
                    "operator": item.operator,
                    "pulse_index": item.pulse_index,
                    "source_sector": item.source_sector,
                    "target_sector": item.target_sector,
                }
                for item in self.interactions
            ),
        }


def translate_ufss_diagrams(
    diagrams,
    *,
    component="custom",
    names=None,
    operator="light_matter",
    amplitudes=None,
    prefactors=None,
    detection="polarization",
):
    """UFSS diagram list -> SolverV10 pathways without an UFSS dependency."""
    diagrams = list(diagrams)
    size = len(diagrams)
    names = list(names) if names is not None else [None] * size
    amplitudes = (
        list(amplitudes) if amplitudes is not None else [1.0] * size
    )
    prefactors = (
        list(prefactors) if prefactors is not None else [None] * size
    )
    if not all(len(items) == size for items in (names, amplitudes, prefactors)):
        raise ValueError("Metadata entries must match the diagrams.")

    pathways = []
    used_names = set()
    for index, (diagram, name, amplitude, prefactor) in enumerate(
        zip(diagrams, names, amplitudes, prefactors), start=1
    ):
        interactions = tuple(
            Interaction(
                label=item[0],
                pulse_index=item[1],
                operator=operator,
            )
            for item in diagram
        )
        base_name = str(name or f"P{index}")
        unique_name = base_name
        suffix = 2
        while unique_name in used_names:
            unique_name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(unique_name)
        pathways.append(
            FrequencyPathway(
                name=unique_name,
                interactions=interactions,
                component=component,
                amplitude=amplitude,
                prefactor=prefactor,
                detection=detection,
            )
        )
    return pathways
