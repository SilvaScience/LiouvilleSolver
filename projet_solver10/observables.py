"""Observable definitions accepted by QuDPy-FDGF.

The module describes final operator contractions, instantaneous GKSL jump
rates, integrated jump counts, and optional action-detection interactions. It
also normalizes the compact public input forms accepted by spectrum
generation into explicit ``ObservableSpec`` instances.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math

from .pathways import Interaction


@dataclass(frozen=True)
class ObservableSpec:
    """Observable settings -> final contraction or GKSL jump measurement.

    Projection interactions enable action-detected observables.
    """

    name: str
    kind: str = "operator"
    operator: str | None = None
    channel: str | None = None
    efficiency: float = 1.0
    time_window: tuple[float, float] | None = None
    n_steps: int = 101
    fourth_interaction: object = None

    def __post_init__(self):
        name = str(self.name).strip()
        if not name:
            raise ValueError("Observable name cannot be empty.")
        kind = str(self.kind).lower().replace("-", "_")
        aliases = {
            "observable": "operator",
            "trace": "operator",
            "mean_jump": "integrated_jump",
            "integrated_rate": "integrated_jump",
        }
        kind = aliases.get(kind, kind)
        if kind not in {"operator", "jump_rate", "integrated_jump"}:
            raise ValueError(
                "kind must be 'operator', 'jump_rate', or "
                "'integrated_jump'."
            )

        operator = self.operator
        if kind == "operator":
            operator = str(operator or name)
            if not operator.strip():
                raise ValueError("An operator observable must have a name.")
        else:
            operator = None if operator is None else str(operator)
            channel = str(self.channel or "").strip()
            if not channel:
                raise ValueError(
                    f"Observable kind {kind!r} must identify a channel."
                )
            object.__setattr__(self, "channel", channel)

        efficiency = float(self.efficiency)
        if not math.isfinite(efficiency) or not 0.0 <= efficiency <= 1.0:
            raise ValueError("efficiency must be between 0 and 1.")

        window = self.time_window
        if kind == "integrated_jump":
            if window is None or len(tuple(window)) != 2:
                raise ValueError(
                    "integrated_jump must provide time_window=(t0, tf)."
                )
            window = tuple(float(value) for value in window)
            if not all(math.isfinite(value) for value in window):
                raise ValueError("time_window must contain finite values.")
            if window[0] < 0.0 or window[1] < window[0]:
                raise ValueError(
                    "time_window must satisfy 0 <= t0 <= tf."
                )
            n_steps = int(self.n_steps)
            if n_steps < 2:
                raise ValueError("n_steps must be at least 2.")
            object.__setattr__(self, "n_steps", n_steps)
        else:
            window = None

        projections = self.fourth_interaction
        if projections is None:
            projections = ()
        elif isinstance(projections, (Interaction, str, Mapping)):
            projections = (projections,)
        else:
            try:
                projections = tuple(projections)
            except TypeError as exc:
                raise TypeError(
                    "fourth_interaction must be an interaction or a "
                    "sequence of interactions."
                ) from exc
        projections = tuple(
            item
            if isinstance(item, Interaction)
            else Interaction(**dict(item))
            if isinstance(item, Mapping)
            else Interaction(str(item))
            for item in projections
        )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "efficiency", efficiency)
        object.__setattr__(self, "time_window", window)
        object.__setattr__(self, "fourth_interaction", projections)

    @property
    def projection_interactions(self):
        """Return alternative interactions applied before detection."""
        return self.fourth_interaction

    @property
    def is_action_detected(self):
        return bool(self.projection_interactions)

    def without_projection(self):
        """Return the final contraction without a projection pulse."""
        if not self.projection_interactions:
            return self
        return replace(self, fourth_interaction=None)

    @classmethod
    def action(
        cls,
        name,
        fourth_interaction,
        *,
        kind="operator",
        operator=None,
        channel=None,
        efficiency=1.0,
        time_window=None,
        n_steps=101,
    ):
        """Projection settings -> action-detected observable."""
        return cls(
            name=name,
            kind=kind,
            operator=operator,
            channel=channel,
            efficiency=efficiency,
            time_window=time_window,
            n_steps=n_steps,
            fourth_interaction=fourth_interaction,
        )

    @classmethod
    def mean_jump(
        cls,
        name,
        channel,
        *,
        time_window,
        efficiency=1.0,
        n_steps=101,
        fourth_interaction=None,
    ):
        """GKSL channel and time window -> integrated mean jump observable."""
        return cls(
            name=name,
            kind="integrated_jump",
            channel=channel,
            time_window=time_window,
            efficiency=efficiency,
            n_steps=n_steps,
            fourth_interaction=fourth_interaction,
        )

    def instantaneous(self):
        """Integrated jump observable -> instantaneous jump rate."""
        if self.kind != "integrated_jump":
            return self
        return replace(self, kind="jump_rate", time_window=None)


def normalize_observables(observables, *, defaults=()):
    """Public observable inputs -> normalized specifications."""
    if observables is None:
        items = list(defaults)
    elif isinstance(observables, (str, ObservableSpec)):
        items = [observables]
    elif isinstance(observables, Mapping):
        items = []
        for alias, value in observables.items():
            alias = str(alias)
            if isinstance(value, ObservableSpec):
                value = replace(value, name=alias)
            elif isinstance(value, str):
                value = ObservableSpec(name=alias, operator=value)
            elif isinstance(value, Mapping):
                payload = dict(value)
                payload.setdefault("name", alias)
                value = ObservableSpec(**payload)
            else:
                raise TypeError(
                    "Each mapped observable must be a name, "
                    "ObservableSpec, or mapping."
                )
            items.append(value)
    else:
        try:
            items = list(observables)
        except TypeError as exc:
            raise TypeError(
                "observables must be a name, sequence, or mapping."
            ) from exc

    normalized = []
    positions = {}
    for item in items:
        if isinstance(item, ObservableSpec):
            spec = item
        elif isinstance(item, str):
            spec = ObservableSpec(name=item)
        elif isinstance(item, Mapping):
            spec = ObservableSpec(**dict(item))
        else:
            raise TypeError(
                "An observable must be a name, ObservableSpec, or mapping."
            )
        if spec.name in positions:
            normalized[positions[spec.name]] = spec
        else:
            positions[spec.name] = len(normalized)
            normalized.append(spec)

    if not normalized:
        raise ValueError("At least one observable is required.")
    return tuple(normalized)
