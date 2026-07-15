# %% [markdown]
# # Etape 2 - Separate constant SOC, dimerisation, and local spin-correlation controls
#
# This file is a clean copy/adaptation of the Step 1 bright/dark model.  The
# physics is still phenomenological: the only new ingredient is an effective
# bright-dark mixing
#
#     V_BD_eff = V0 + lambda_delta * delta + lambda_C * C1
#
# The goal is to run one case at a time and compare the third-order rephasing
# 1Q response against the Step 1 baseline.  No parameter sweep is executed here.

# %%
from pathlib import Path
import json
import os
import sys

import numpy as np

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "SolverV8").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SolverV8 import (
    LiouvilleSpectroscopySolver,
    SpectroscopyPlotter,
    standard_nq_protocol,
)

RESULT_ROOT = Path(os.environ["ZOOM_RESULT_ROOT"])
DATA_DIR = RESULT_ROOT / "Data"
FIGURES_DIR = RESULT_ROOT / "Figures"
SUMMARIES_DIR = RESULT_ROOT / "Summaries"

for directory in (DATA_DIR, FIGURES_DIR, SUMMARIES_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Parameters
#
# Use `active_case_key` to choose one scenario.  The three default scenarios are
# tuned so that the added dimerisation and spin-correlation contributions are
# comparable in size to the Step 1 `V0 = 0.01` control.

# %%
base_model_params = {
    # Orbital sector: |g>, |D>, |B>
    "Delta_dark": 0.90,
    "Delta_Bright": 1.00,
    "mu_B": 1.0,
    "mu_D": 0.0,

    # Effective bright-dark mixing controls
    "V0": 0.01,
    "lambda_delta": 0.0,
    "lambda_C": 0.0,
    "delta_source": "external",  # "external" or "spin_peierls_mean_field"
    "delta_external": 0.0,
    "C1": 0.0,

    # Optional dynamic spin-Peierls mode, off by default for this control test
    "N_k": 1,
    "n_bosons": 1,
    "T": 1.0,
    "T_SP_0": 14.0,
    "B": 0.0,
    "J": 1.0,
    "delta_0": 0.01,
    "beta": 0.5,
    "spin_mode_scale": 0.0,
    "Lambda_SOC": 0.0,

    # k-dependent coupling shape, only relevant if N_k > 1 and Lambda_SOC != 0
    "a_dimer": 1.0,
    "alpha_dimerisation": 0.01,
    "parity": "odd",

    # Dissipation
    "gamma_orb": 0.0,
    "gamma_spin": 0.0,
}

scenario_params = {
    "soc_constant": {
        "label": "step2_soc_constant",
        "V0": 0.01,
        "lambda_delta": 0.0,
        "lambda_C": 0.0,
        "delta_external": 0.0,
        "C1": 0.0,
    },
    "dimerisation_control": {
        "label": "step2_dimerisation_control",
        "V0": 0.01,
        "lambda_delta": 0.20,
        "lambda_C": 0.0,
        "delta_external": 0.05,
        "C1": 0.0,
    },
    "spin_correlation_control": {
        "label": "step2_spin_correlation_control",
        "V0": 0.01,
        "lambda_delta": 0.0,
        "lambda_C": -0.02257336343115124,
        "delta_external": 0.0,
        "C1": -0.443,
    },
}

active_case_key = os.environ.get("STEP2_ACTIVE_CASE", "soc_constant")
if active_case_key not in scenario_params:
    raise ValueError(
        f"Unknown STEP2_ACTIVE_CASE={active_case_key!r}. "
        f"Available cases: {tuple(scenario_params)}"
    )
active_params = {
    **base_model_params,
    **scenario_params[active_case_key],
}

solver_params = {
    "T": 0.0,
    "Eta": 0.001,
    "backend": "dense",
    "parallel_backend": "threading",
    "n_jobs": -1,
}


# %% [markdown]
# ## Model Builder

# %%
def spin_peierls_delta(T, B, T_SP_0, delta_0, beta):
    alpha_field = 0.004
    T_SP = max(0.0, T_SP_0 * (1.0 - alpha_field * B**2))
    if T_SP > 0.0 and T < T_SP:
        return delta_0 * (1.0 - T / T_SP) ** beta
    return 0.0


def resolve_delta(params):
    if params["delta_source"] == "spin_peierls_mean_field":
        return spin_peierls_delta(
            params["T"],
            params["B"],
            params["T_SP_0"],
            params["delta_0"],
            params["beta"],
        )
    return float(params["delta_external"])


def effective_bright_dark_mixing(params, delta):
    return (
        float(params["V0"])
        + float(params["lambda_delta"]) * float(delta)
        + float(params["lambda_C"]) * float(params["C1"])
    )


def spin_peierls_dispersion(k, params, delta):
    gap = 2.0 * params["J"] * delta ** (2.0 / 3.0) if delta > 0.0 else 0.0
    velocity = np.pi * params["J"] / 2.0
    omega_k = np.sqrt(gap**2 + (velocity * np.sin(k)) ** 2)
    return omega_k, gap


def k_coupling(k, delta, alpha_dimerisation, a_dimer=1.0, parity="odd"):
    phase = k * a_dimer / 2.0
    eta = delta * alpha_dimerisation
    if parity == "odd":
        return np.sin(phase)
    if parity == "even":
        return np.cos(phase)
    return np.sin(phase) + eta * np.cos(phase)


def lowering_operator(n):
    op = np.zeros((n, n), dtype=complex)
    for upper in range(1, n):
        op[upper - 1, upper] = np.sqrt(upper)
    return op


def build_soc_bright_dark_model(params):
    N_k = int(params["N_k"])
    n_bosons = int(params["n_bosons"])
    if N_k == 1:
        k_array = np.array([0.0])
        k_weights = np.ones(1)
    else:
        k_array = np.linspace(-np.pi, np.pi, N_k, endpoint=False)
        k_weights = np.ones(N_k) / N_k

    ket_g, ket_d, ket_b = 0, 1, 2
    H_orb = np.zeros((3, 3), dtype=complex)
    H_orb[ket_d, ket_d] = params["Delta_dark"]
    H_orb[ket_b, ket_b] = params["Delta_Bright"]

    L_bd = np.zeros((3, 3), dtype=complex)
    L_bd[ket_d, ket_b] = 1.0
    L_bd[ket_b, ket_d] = 1.0

    mu_orb = np.zeros((3, 3), dtype=complex)
    mu_orb[ket_g, ket_b] = params["mu_B"]
    mu_orb[ket_b, ket_g] = params["mu_B"]
    mu_orb[ket_g, ket_d] = params["mu_D"]
    mu_orb[ket_d, ket_g] = params["mu_D"]

    a = lowering_operator(n_bosons)
    adag = a.conj().T
    n_op = adag @ a
    x_op = a + adag

    I_orb = np.eye(3, dtype=complex)
    I_spin = np.eye(n_bosons, dtype=complex)
    dim = 3 * n_bosons

    H_stack = np.zeros((N_k, dim, dim), dtype=complex)
    mu_stack = np.zeros_like(H_stack)
    rho0 = np.zeros_like(H_stack)

    delta = resolve_delta(params)
    V_BD_eff = effective_bright_dark_mixing(params, delta)

    for i_k, k in enumerate(k_array):
        omega_k, gap = spin_peierls_dispersion(k, params, delta)
        V_k = k_coupling(
            k,
            delta,
            params["alpha_dimerisation"],
            a_dimer=params["a_dimer"],
            parity=params["parity"],
        )

        H_local = np.kron(H_orb, I_spin)
        H_spin = params["spin_mode_scale"] * omega_k * np.kron(I_orb, n_op)
        H_static_mix = V_BD_eff * np.kron(L_bd, I_spin)
        H_mode_mix = params["Lambda_SOC"] * V_k * np.kron(L_bd, x_op)

        H_stack[i_k] = H_local + H_spin + H_static_mix + H_mode_mix
        mu_stack[i_k] = np.kron(mu_orb, I_spin)
        rho0[i_k, 0, 0] = 1.0

    c_ops = []
    if params["gamma_spin"]:
        c_ops.append((
            np.repeat(np.kron(I_orb, a)[None, :, :], N_k, axis=0),
            params["gamma_spin"],
        ))
    if params["gamma_orb"]:
        C_bg = np.zeros((3, 3), dtype=complex)
        C_bg[ket_g, ket_b] = 1.0
        C_dg = np.zeros((3, 3), dtype=complex)
        C_dg[ket_g, ket_d] = 1.0
        c_ops.append((
            np.repeat(np.kron(C_bg, I_spin)[None, :, :], N_k, axis=0),
            params["gamma_orb"],
        ))
        c_ops.append((
            np.repeat(np.kron(C_dg, I_spin)[None, :, :], N_k, axis=0),
            params["gamma_orb"],
        ))

    metadata = {
        "k_array": k_array,
        "k_weights": k_weights,
        "delta": delta,
        "C1": float(params["C1"]),
        "V_BD_eff": V_BD_eff,
        "spin_gap": spin_peierls_dispersion(0.0, params, delta)[1],
        "dim": dim,
    }
    return H_stack, mu_stack, c_ops, rho0, metadata


# %% [markdown]
# ## Solver Setup

# %%
def make_solver(params):
    H, mu, c_ops, rho0, meta = build_soc_bright_dark_model(params)

    solver = LiouvilleSpectroscopySolver(solver_params)
    solver.feed_model(
        H,
        mu,
        c_ops_raw=c_ops,
        initial_density_matrix=rho0,
        density_matrix_basis="site",
    )
    return solver, meta


solver, meta = make_solver(active_params)
print(active_params["label"])
print("Hilbert dimension:", meta["dim"])
print("N_k:", len(meta["k_array"]))
print("delta:", meta["delta"])
print("C1:", meta["C1"])
print("V_BD_eff:", meta["V_BD_eff"])
print("Hamiltonian dimensions:", solver.H_eigen.shape)


# %% [markdown]
# ## Third-Order 1Q Pathways With NQ Protocol

# %%
arrival_times = [0.0, 100.0, 200.0]
pathways = solver.configure_standard_2d_pathways_with_ufss(arrival_times)

protocol = standard_nq_protocol(
    order=1,
    nq_interval=1,
    detection_interval=3,
    n_interactions=3,
    nq_axis="omega1",
    detection_axis="omega3",
)

rephasing_pathways = solver.get_pathways("rephasing")
unrephasing_pathways = solver.get_pathways("unrephasing")

[(p.name, p.component, p.interactions, p.coherence_orders) for p in pathways]


# %% [markdown]
# ## Spectrum Calculation
#
# Run one active case at a time.  Change `active_case_key` above, restart the
# kernel, and rerun when producing the actual results.

# %%
N_w = int(os.environ.get("ZOOM_N", "161"))
zoom_center = float(os.environ["ZOOM_CENTER_EV"])
zoom_half_width = float(os.environ.get("ZOOM_HALF_WIDTH_EV", "0.04"))
omega1_rephasing = np.linspace(-zoom_center - zoom_half_width, -zoom_center + zoom_half_width, N_w)
omega1_unrephasing = np.linspace(zoom_center - zoom_half_width, zoom_center + zoom_half_width, N_w)
omega1 = np.concatenate([omega1_rephasing, omega1_unrephasing])
omega3 = np.linspace(zoom_center - zoom_half_width, zoom_center + zoom_half_width, N_w)
tau2 = 3.0

result = solver.generate_NQ_spectrum(
    1,
    protocol,
    axes={"omega1": omega1, "omega3": omega3},
    delays={"t2": tau2},
    pathways=pathways,
    k_array=meta["k_array"],
    k_weights=meta["k_weights"],
)

print("Components:", tuple(result.components))
print("Pathways:", tuple(result.pathways))


# %% [markdown]
# ## Save And Plot

# %%
def safe_float_label(value):
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


save_pdf = False
plot_pathways = [p.name for p in rephasing_pathways]
plot_totals = ["rephasing"]
plot_view = "real"
plot_normalization = "individual"

case_label = (
    f"{active_params['label']}"
    f"_Eta_{safe_float_label(solver_params['Eta'])}"
    f"_Veff_{safe_float_label(meta['V_BD_eff'])}"
    f"_delta_{safe_float_label(meta['delta'])}"
    f"_C1_{safe_float_label(meta['C1'])}"
)

figure_directory = FIGURES_DIR / active_params["label"]
figure_directory.mkdir(parents=True, exist_ok=True)
data_file = DATA_DIR / f"{case_label}_S_data.npz"
summary_txt = SUMMARIES_DIR / f"{case_label}_results.txt"
summary_json = SUMMARIES_DIR / f"{case_label}_summary.json"

plotter = SpectroscopyPlotter(detection_phase=np.pi / 2)
plot_result = plotter.plot_pathways_multiorder(
    result,
    pathways=plot_pathways,
    totals=plot_totals,
    view=plot_view,
    normalization=plot_normalization,
    axis_labels={
        "omega1": "1Q frequency (eV)",
        "omega3": "Detection frequency (eV)",
    },
    include_diagrams=False,
    display_diagrams=False,
    save_pdf=save_pdf,
    output_directory=figure_directory if save_pdf else None,
    spectrum_pdf_name=f"{case_label}.pdf",
    show=False,
)

print("Plotted panels:", plot_result.panel_names)

if True:
    spectrum_data = {
        "omega1": omega1,
        "omega3": omega3,
    }
    for name, matrix in result.components.items():
        spectrum_data[f"S_component_{name}"] = matrix
    for name, matrix in result.pathways.items():
        spectrum_data[f"S_pathway_{name}"] = matrix
    np.savez_compressed(data_file, **spectrum_data)

    summary = {
        "test": "etape_2",
        "run_label": case_label,
        "active_case_key": active_case_key,
        "model_parameters": active_params,
        "solver_parameters": solver_params,
        "derived": {
            "delta": float(meta["delta"]),
            "C1": float(meta["C1"]),
            "V_BD_eff": float(meta["V_BD_eff"]),
            "spin_gap": float(meta["spin_gap"]),
            "N_k": int(len(meta["k_array"])),
            "weight_sum": float(np.sum(meta["k_weights"])),
        },
        "spectrum_parameters": {
            "N_w": int(N_w),
            "tau2": float(tau2),
            "omega1_min": float(omega1[0]),
            "omega1_max": float(omega1[-1]),
            "omega3_min": float(omega3[0]),
            "omega3_max": float(omega3[-1]),
            "plot_pathways": plot_pathways,
            "plot_totals": plot_totals,
            "view": plot_view,
            "normalization": plot_normalization,
        },
        "files": {
            "spectrum_pdf": str(plot_result.spectrum_pdf),
            "data_npz": str(data_file),
            "summary_txt": str(summary_txt),
        },
        "validation_status": "PENDING PHASE 2",
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"Test: etape_2")
    lines.append(f"Run label: {case_label}")
    lines.append(f"Active case: {active_case_key}")
    lines.append("")
    lines.append("Effective bright-dark mixing")
    lines.append("----------------------------")
    lines.append("V_BD_eff = V0 + lambda_delta * delta + lambda_C * C1")
    lines.append(f"V0: {active_params['V0']}")
    lines.append(f"lambda_delta: {active_params['lambda_delta']}")
    lines.append(f"delta: {meta['delta']}")
    lines.append(f"lambda_C: {active_params['lambda_C']}")
    lines.append(f"C1: {meta['C1']}")
    lines.append(f"V_BD_eff: {meta['V_BD_eff']}")
    lines.append("")
    lines.append("Model parameters")
    lines.append("----------------")
    for key, value in active_params.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Solver parameters")
    lines.append("-----------------")
    for key, value in solver_params.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Spectrum parameters")
    lines.append("-------------------")
    lines.append(f"N_w: {int(N_w)}")
    lines.append(f"tau2: {float(tau2)}")
    lines.append(f"omega1_min: {float(omega1[0])}")
    lines.append(f"omega1_max: {float(omega1[-1])}")
    lines.append(f"omega3_min: {float(omega3[0])}")
    lines.append(f"omega3_max: {float(omega3[-1])}")
    lines.append(f"plot_pathways: {plot_pathways}")
    lines.append(f"plot_totals: {plot_totals}")
    lines.append(f"view: {plot_view}")
    lines.append(f"normalization: {plot_normalization}")
    lines.append("")
    lines.append("Files")
    lines.append("-----")
    lines.append(f"Spectrum PDF: {plot_result.spectrum_pdf}")
    lines.append(f"Spectrum data: {data_file}")
    lines.append(f"Summary JSON: {summary_json}")
    lines.append("")
    lines.append("Validation status: PENDING PHASE 2")
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Spectrum PDF:", plot_result.spectrum_pdf)
    print("Spectrum data:", data_file)
    print("Summary TXT:", summary_txt)
    print("Summary JSON:", summary_json)
