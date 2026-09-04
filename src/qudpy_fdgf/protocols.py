"""Propagation protocols independent of the physical model.

Protocols name each interval, select its time, frequency, or identity domain,
and optionally constrain coherence order. They validate pathway topology and
coordinate inputs before a backend begins propagation, and provide a helper
for standard two-frequency NQ experiments.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PropagationInterval:
    """Light-matter interaction -> propagation interval."""

    name: str
    domain: str
    coherence_order: int | None = None
    eta: float | None = None

    def __post_init__(self):
        name = str(self.name)
        domain = str(self.domain).lower()
        if domain not in {"time", "frequency", "identity"}:
            raise ValueError(
                "domain must be 'time', 'frequency', or 'identity'."
            )
        if domain != "identity" and not name:
            raise ValueError("A propagated interval must have a name.")
        if self.eta is not None and float(self.eta) < 0:
            raise ValueError("eta must be non-negative.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "domain", domain)
        if self.coherence_order is not None:
            object.__setattr__(
                self, "coherence_order", int(self.coherence_order)
            )
        if self.eta is not None:
            object.__setattr__(self, "eta", float(self.eta))


@dataclass(frozen=True)
class SpectroscopyProtocol:
    """Ordered propagation intervals -> pathway protocol."""

    intervals: tuple
    name: str = "custom"

    def __post_init__(self):
        intervals = tuple(
            item
            if isinstance(item, PropagationInterval)
            else PropagationInterval(**item)
            for item in self.intervals
        )
        if not intervals:
            raise ValueError("A protocol must contain at least one interval.")
        names = [
            item.name for item in intervals if item.domain != "identity"
        ]
        if len(names) != len(set(names)):
            raise ValueError("Interval names must be unique.")
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "name", str(self.name))

    @property
    def coordinate_names(self):
        return tuple(
            item.name for item in self.intervals if item.domain != "identity"
        )

    @property
    def frequency_axis_names(self):
        return tuple(
            item.name
            for item in self.intervals
            if item.domain == "frequency"
        )

    @property
    def time_axis_names(self):
        return tuple(
            item.name for item in self.intervals if item.domain == "time"
        )

    def validate_pathway(self, pathway):
        """Pathway -> validation of length and coherence orders."""
        if len(pathway.interactions) != len(self.intervals):
            raise ValueError(
                f"Protocol {self.name!r} has {len(self.intervals)} intervals, "
                f"but pathway {pathway.name!r} has "
                f"{len(pathway.interactions)} interactions."
            )
        for index, (interval, actual) in enumerate(
            zip(self.intervals, pathway.coherence_orders), start=1
        ):
            expected = interval.coherence_order
            if expected is not None and abs(expected) != abs(actual):
                raise ValueError(
                    f"Incompatible coherence order at step {index}: "
                    f"|q|={abs(actual)}, expected {abs(expected)}."
                )

    def validate_coordinates(self, coordinates):
        """Coordinates -> validation of all required interval values."""
        missing = sorted(set(self.coordinate_names).difference(coordinates))
        if missing:
            raise KeyError(f"Missing coordinates: {missing}")


def standard_nq_protocol(
    order,
    nq_interval,
    detection_interval,
    *,
    n_interactions=3,
    nq_axis=None,
    detection_axis="omega3",
):
    """NQ settings -> protocol with two frequency axes."""
    order = abs(int(order))
    nq_index = int(nq_interval) - 1
    detection_index = int(detection_interval) - 1
    n_interactions = int(n_interactions)
    if n_interactions < 1:
        raise ValueError("n_interactions must be positive.")
    if not 0 <= nq_index < n_interactions:
        raise ValueError("nq_interval is outside the protocol.")
    if not 0 <= detection_index < n_interactions:
        raise ValueError("detection_interval is outside the protocol.")
    if nq_index == detection_index:
        raise ValueError("The two frequency intervals must differ.")

    intervals = []
    for index in range(n_interactions):
        if index == nq_index:
            intervals.append(
                PropagationInterval(
                    nq_axis or f"omega{order}q",
                    "frequency",
                    order,
                )
            )
        elif index == detection_index:
            intervals.append(
                PropagationInterval(detection_axis, "frequency")
            )
        else:
            intervals.append(
                PropagationInterval(f"t{index + 1}", "time")
            )
    return SpectroscopyProtocol(tuple(intervals), f"standard_{order}q")
