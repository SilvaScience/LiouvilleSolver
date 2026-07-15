"""Standard spectroscopy protocol builders."""

from .models import PropagationInterval, SpectroscopyProtocol


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

