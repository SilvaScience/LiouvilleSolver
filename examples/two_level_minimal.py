"""Minimal two-level spectroscopy example.

The script builds a site-basis Hamiltonian and transition dipole, adapts them
to the QuDPy-FDGF model contract, and computes linear absorption. It then
evaluates rephasing and non-rephasing third-order pathways on their physical
frequency quadrants and renders the results with ``SpectroscopyPlotter``.
"""

from pathlib import Path
import sys

import numpy as np


def _add_project_root_to_path():
    """Example path -> project root added to ``sys.path``."""
    example_file = Path(__file__).resolve()
    candidates = (example_file.parent, *example_file.parents)
    for candidate in candidates:
        package_entry = candidate / "projet_solver10.py"
        if package_entry.is_file():
            project_root = str(candidate)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            return
    raise RuntimeError(
        "Could not find a parent directory containing projet_solver10.py."
    )


_add_project_root_to_path()

from projet_solver10 import (
    EigenbasisKModel,
    SpectroscopyPlotter,
    SpectroscopySolver,
)
from projet_solver10.pathways import FrequencyPathway
from projet_solver10.protocols import standard_nq_protocol, PropagationInterval, SpectroscopyProtocol

# --- 1. Build site-basis Hamiltonian and dipole matrices ---
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

H_site = 1.0 * sigma_z      # 2 eV gap between |g> and |e>
mu_site = 0.5 * sigma_x     # Transition dipole

# --- 2. Adapt the matrices to SectorModel ---
model = EigenbasisKModel(H_site, mu_site)

# --- 3. Select a backend and initialize the solver ---
solver = SpectroscopySolver(backend="dense", eta=0.0005)

# --- 4. Load the model with its default ground state ---
solver.feed_model(model)

# --- 5. Define the single-interaction linear-absorption pathway ---
pathway = FrequencyPathway(
    name="linear",
    interactions=[{"label": "Ku"}],
    component="linear",
)
solver.set_pathways([pathway])

# --- 6. Define one frequency-domain interval ---
protocol = SpectroscopyProtocol(
    intervals=[PropagationInterval("omega1", "frequency")]
)

# --- 7. Define the frequency axis ---
axes = {"omega1": np.linspace(-4.0, 4.0, 801)}

# --- 8. Calculate linear absorption ---
result = solver.generate_spectrum(protocol, axes)

# --- 9. Extract the linear response ---
omega1 = result.axis_values[0]
signal = result.pathways["linear"]

# --- 10. Define third-order 1Q pathways ---
# A two-level system has no biexciton manifold and therefore no ESA pathway.
# Rephasing pathways R1 and R2 have q=(-1, 0, +1) over (t1, t2, t3).
pathway_r1 = FrequencyPathway(
    name="R1",
    interactions=[
        {"label": "Bu", "pulse_index": 0},
        {"label": "Ku", "pulse_index": 1},
        {"label": "Bd", "pulse_index": 2},
    ],
    component="rephasing",
)
pathway_r2 = FrequencyPathway(
    name="R2",
    interactions=[
        {"label": "Bu", "pulse_index": 0},
        {"label": "Bd", "pulse_index": 1},
        {"label": "Ku", "pulse_index": 2},
    ],
    component="rephasing",
)
pathway_nonrephasing = FrequencyPathway(
    name="R4_nonrephasing",
    interactions=[
        {"label": "Ku", "pulse_index": 0},
        {"label": "Bu", "pulse_index": 1},
        {"label": "Bd", "pulse_index": 2},
    ],
    component="nonrephasing",
)
print("R1 rephasing coherence order (q):", pathway_r1.coherence_orders)
print("R2 rephasing coherence order (q):", pathway_r2.coherence_orders)
print("Non-rephasing coherence order (q):", pathway_nonrephasing.coherence_orders)

# --- 11. Use frequency axes for t1 and t3; fix waiting time t2 to zero ---
protocol_1q = standard_nq_protocol(
    order=1, nq_interval=1, detection_interval=3, n_interactions=3,
    nq_axis="omega1q",
)

# Rephasing: q=-1 during t1 -> negative omega1q, positive omega3.
# Non-rephasing: q=+1 during t1 -> positive omega1q and omega3.
# Distinct physical quadrants require separate axes and solver calls.
axes_rephasing = {
    "omega1q": np.linspace(-4.0, 0.0, 161),
    "omega3": np.linspace(0.0, 4.0, 161),
}
axes_nonrephasing = {
    "omega1q": np.linspace(0.0, 4.0, 161),
    "omega3": np.linspace(0.0, 4.0, 161),
}

# --- 12. Calculate each pathway on its physical quadrant ---
result_rephasing = solver.generate_spectrum(
    protocol_1q, axes_rephasing,
    fixed_coordinates={"t2": 0.0}, pathways=[pathway_r1, pathway_r2],
)
result_nonrephasing = solver.generate_spectrum(
    protocol_1q, axes_nonrephasing,
    fixed_coordinates={"t2": 0.0}, pathways=[pathway_nonrephasing],
)
rephasing = result_rephasing.components["rephasing"]
nonrephasing = result_nonrephasing.components["unrephasing"]  # "nonrephasing" -> "unrephasing"


def _print_peak(label, data, omega1q_axis, omega3_axis):
    index = np.unravel_index(np.argmax(np.abs(data)), data.shape)
    print(
        f"{label}: peak |S|={np.abs(data[index]):.4f} at "
        f"(omega1q, omega3) = "
        f"({omega1q_axis[index[0]]:.3f}, {omega3_axis[index[1]]:.3f}) eV"
    )


print()
print("--- Individual pathways ---")
_print_peak(
    "R1_rephasing", result_rephasing.pathways["R1"],
    result_rephasing.axis_values[0], result_rephasing.axis_values[1],
)
_print_peak(
    "R2_rephasing", result_rephasing.pathways["R2"],
    result_rephasing.axis_values[0], result_rephasing.axis_values[1],
)
_print_peak(
    "R4_nonrephasing", result_nonrephasing.pathways["R4_nonrephasing"],
    result_nonrephasing.axis_values[0], result_nonrephasing.axis_values[1],
)
print()
print("--- Rephasing / non-rephasing components ---")
_print_peak(
    "Rephasing", rephasing,
    result_rephasing.axis_values[0], result_rephasing.axis_values[1],
)
_print_peak(
    "Non-rephasing", nonrephasing,
    result_nonrephasing.axis_values[0], result_nonrephasing.axis_values[1],
)

# --- 13. Plot with the V10 plotter ---
# Figure 2 convention: real is absorptive and imaginary is dispersive.
plotter = SpectroscopyPlotter(detection_phase=0)
plotter_2d = SpectroscopyPlotter(detection_phase=np.pi / 2)
plotter.plot_1d(
    signal,
    w=omega1,
    params={
        "view": "all",
        "title": "Linear absorption",
        "xlabel": r"$\omega_1$ (eV)",
        "ylabel": "Signal",
        "reference_positions": (2.0,),
        "show": True,
    },
)

plotter_2d.plot_real_imag_abs(
    result_rephasing,
    {
        "pathways": ["R1"],
        "totals": False,
        "diagonals": "auto",
        "labels": (r"$\omega_3$ (eV)", r"$\omega_{1q}$ (eV)"),
        "title": "Rephasing R1: absorptive / dispersive / abs",
        "style": {
            "cmap": "RdYlBu_r",
            "abs_cmap": "magma",
            "levels": 20,
            "contour_lines": False,
        },
        "show": True,
    },
)

plotter_2d.plot_spectrum_result(
    result_nonrephasing,
    {
        "source": "components",
        "names": ["unrephasing"],
        "totals": True,
        "view": "real",
        "diagonals": "auto",
        "labels": (r"$\omega_3$ (eV)", r"$\omega_{1q}$ (eV)"),
        "title": "Non-rephasing (R4): absorptive",
        "style": {
            "abs_cmap": "magma",
            "levels": 20,
            "contour_lines": True,
        },
        "show": True,
    },
)
