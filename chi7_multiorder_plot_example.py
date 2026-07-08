"""Seventh-order multi-pathway plotting example based on Test 11."""

from pathlib import Path

import numpy as np

from SolverV8 import (
    LiouvilleSpectroscopySolver,
    SpectroscopyPlotter,
    standard_nq_protocol,
)


model_params = {
    "omega_h": 1.53,
    "omega_l": 1.58,
    "J": 0.010,
    "mu_h": 1.0,
    "mu_l": 0.3,
    "kappa": 0.001,
    "U_hh": -0.020,
    "U_ll": -0.020,
    "U_hl": 0.01,
    "max_manifold": 4,
}

solver_params = {
    "T": 0.0,
    "Eta": 0.005,
    "backend": "dense",
    "parallel_backend": "threading",
    "n_jobs": -1,
}


def build_bosonic_triexciton_model(params):
    maximum_manifold = int(params["max_manifold"])
    basis = tuple(
        (n_h, total - n_h)
        for total in range(maximum_manifold + 1)
        for n_h in range(total, -1, -1)
    )
    index = {state: i for i, state in enumerate(basis)}

    b_h = np.zeros((len(basis), len(basis)), dtype=complex)
    b_l = np.zeros_like(b_h)
    for upper_index, (n_h_value, n_l_value) in enumerate(basis):
        if n_h_value:
            b_h[index[(n_h_value - 1, n_l_value)], upper_index] = np.sqrt(
                n_h_value
            )
        if n_l_value:
            b_l[index[(n_h_value, n_l_value - 1)], upper_index] = np.sqrt(
                n_l_value
            )

    n_h = b_h.conj().T @ b_h
    n_l = b_l.conj().T @ b_l
    identity = np.eye(len(basis))
    H = (
        params["omega_h"] * n_h
        + params["omega_l"] * n_l
        + params["J"] * (b_h.conj().T @ b_l + b_l.conj().T @ b_h)
        + 0.5 * params["U_hh"] * n_h @ (n_h - identity)
        + 0.5 * params["U_ll"] * n_l @ (n_l - identity)
        + params["U_hl"] * n_h @ n_l
    )
    mu = (
        params["mu_h"] * (b_h + b_h.conj().T)
        + params["mu_l"] * (b_l + b_l.conj().T)
    )
    c_ops = [(b_h, params["kappa"]), (b_l, params["kappa"])]
    rho0 = np.zeros_like(H)
    rho0[0, 0] = 1.0
    return H, mu, c_ops, rho0, basis


H, mu, c_ops, rho0, basis = build_bosonic_triexciton_model(model_params)
solver = LiouvilleSpectroscopySolver(solver_params)
solver.feed_model(
    H,
    mu,
    c_ops_raw=c_ops,
    initial_density_matrix=rho0,
    density_matrix_basis="site",
)

pathways = solver.generate_pathways_with_ufss(
    "--+++",
    maximum_manifold=2,
    component="chi5_2q",
)

protocol = standard_nq_protocol(
    order=2,
    nq_interval=2,
    detection_interval=5,
    n_interactions=5,
    nq_axis="omega_2q",
    detection_axis="omega_emit",
)




omega_2q   = np.linspace(-3.18,-2.88, 50)
omega_emit = np.linspace(1.38, 1.68, 50)
delays = {
    interval.name: 0.0
    for interval in protocol.intervals
    if interval.domain == "time"
}

result = solver.generate_NQ_spectrum(
    2 ,
    protocol,
    axes={"omega_2q": omega_2q, "omega_emit": omega_emit},
    delays=delays,
    pathways=pathways,
)

save_pdf = True
output_directory = (
    Path(__file__).resolve().parent
    / "Result_Test"
    / "Multiorder_pathway_plot"
)

plotter = SpectroscopyPlotter(detection_phase=0)
plot_result = plotter.plot_pathways_multiorder(
    result,
    pathways="all",
    totals=["2Q"],
    view="real",
    normalization="individual",
    axis_labels={
        "omega_2q": "2Q energy (eV)",
        "omega_emit": "Emission energy (eV)",
    },
    include_diagrams=False,
    display_diagrams=False,
    save_pdf=save_pdf,
    output_directory=output_directory if save_pdf else None,
    spectrum_pdf_name="chi5_pathways_and_total_detection_phase_0.pdf",
    show=True,
)

print("Plotted panels:", plot_result.panel_names)
print("Matching UFSS diagrams:", tuple(plot_result.diagrams))
if save_pdf:
    print("Spectrum PDF:", plot_result.spectrum_pdf)
    print("Diagram PDFs:", plot_result.diagram_paths)
