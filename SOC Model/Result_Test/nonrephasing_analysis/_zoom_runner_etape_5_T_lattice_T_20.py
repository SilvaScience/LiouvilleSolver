# %% [markdown]
# # Etape 5 - Temperature and field scans
#
# Step 4 validated four discrete trend controls.  Step 5 turns that logic into
# scans:
#
# - sweep T across T_SP at B=0 for the lattice-dynamic channel;
# - sweep B at low T for field suppression of the lattice channel;
# - sweep T for the static C1 channel as a non-unique persistent control.

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

# %%
def spin_peierls_transition_temperature(B, T_SP_0, alpha_field):
    return max(0.0, float(T_SP_0) * (1.0 - float(alpha_field) * float(B) ** 2))


def spin_peierls_delta(T, B, T_SP_0, delta_0, beta, alpha_field):
    T_SP = spin_peierls_transition_temperature(B, T_SP_0, alpha_field)
    if T_SP > 0.0 and T < T_SP:
        return float(delta_0) * (1.0 - float(T) / T_SP) ** float(beta)
    return 0.0


def spin_correlation_proxy(T, B, T_SP_0, C1_lowT, C1_highT, C1_field_scale):
    thermal_weight = 1.0 / (1.0 + float(T) / float(T_SP_0))
    field_weight = 1.0 / (1.0 + float(C1_field_scale) * float(B) ** 2)
    return float(C1_highT) + (float(C1_lowT) - float(C1_highT)) * thermal_weight * field_weight


def safe_float_label(value):
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


trend_params = {
    "T_SP_0": 14.0,
    "delta_0": 0.01,
    "beta": 0.5,
    "alpha_field": 0.004,
    "T_ref": 7.0,
    "B_ref": 0.0,
    "C1_lowT": -0.443,
    "C1_highT": -0.25,
    "C1_field_scale": 0.0015,
}

delta_ref = spin_peierls_delta(
    trend_params["T_ref"],
    trend_params["B_ref"],
    trend_params["T_SP_0"],
    trend_params["delta_0"],
    trend_params["beta"],
    trend_params["alpha_field"],
)
C1_highT_ref = spin_correlation_proxy(
    20.0,
    0.0,
    trend_params["T_SP_0"],
    trend_params["C1_lowT"],
    trend_params["C1_highT"],
    trend_params["C1_field_scale"],
)

matched_lambda_delta = 0.01 / delta_ref
matched_lambda_C_highT = 0.01 / C1_highT_ref

base_model_params = {
    "Delta_dark": 0.90,
    "Delta_Bright": 1.00,
    "mu_B": 1.0,
    "mu_D": 0.0,
    "V0": 0.01,
    "lambda_delta": 0.0,
    "lambda_C": 0.0,
    "n_bosons": 3,
    "omega_Q": 0.035,
    "g_Q0": 0.010,
    "g_Q_source": "off",
    "g_Q_external": 0.0,
    **trend_params,
    "T": 7.0,
    "B": 0.0,
    "N_k": 1,
    "gamma_orb": 0.0,
    "gamma_phonon": 0.0,
}


def build_scan_cases():
    cases = {}
    for T in (4.0, 7.0, 10.0, 13.0, 14.0, 16.0, 20.0):
        key = f"T_lattice_T_{safe_float_label(T)}"
        cases[key] = {
            "label": f"step5_T_lattice_T_{safe_float_label(T)}",
            "scan_family": "temperature_lattice_dynamic",
            "scan_variable": "T",
            "scan_value": T,
            "channel": "lattice_dynamic",
            "T": T,
            "B": 0.0,
            "lambda_delta": matched_lambda_delta,
            "lambda_C": 0.0,
            "g_Q_source": "delta_scaled",
        }
    for B in (0.0, 5.0, 10.0, 13.0, 15.0):
        key = f"B_lattice_B_{safe_float_label(B)}"
        cases[key] = {
            "label": f"step5_B_lattice_B_{safe_float_label(B)}",
            "scan_family": "field_lattice_dynamic",
            "scan_variable": "B",
            "scan_value": B,
            "channel": "lattice_dynamic",
            "T": 7.0,
            "B": B,
            "lambda_delta": matched_lambda_delta,
            "lambda_C": 0.0,
            "g_Q_source": "delta_scaled",
        }
    for T in (4.0, 7.0, 10.0, 13.0, 14.0, 16.0, 20.0):
        key = f"T_C1_static_T_{safe_float_label(T)}"
        cases[key] = {
            "label": f"step5_T_C1_static_T_{safe_float_label(T)}",
            "scan_family": "temperature_C1_static",
            "scan_variable": "T",
            "scan_value": T,
            "channel": "C1_static",
            "T": T,
            "B": 0.0,
            "lambda_delta": 0.0,
            "lambda_C": matched_lambda_C_highT,
            "g_Q_source": "off",
        }
    return cases


scenario_params = build_scan_cases()
active_case_key = os.environ.get("STEP5_ACTIVE_CASE", "T_lattice_T_7")
if active_case_key not in scenario_params:
    raise ValueError(
        f"Unknown STEP5_ACTIVE_CASE={active_case_key!r}. "
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
def resolve_trends(params):
    T_SP = spin_peierls_transition_temperature(
        params["B"],
        params["T_SP_0"],
        params["alpha_field"],
    )
    delta = spin_peierls_delta(
        params["T"],
        params["B"],
        params["T_SP_0"],
        params["delta_0"],
        params["beta"],
        params["alpha_field"],
    )
    C1 = spin_correlation_proxy(
        params["T"],
        params["B"],
        params["T_SP_0"],
        params["C1_lowT"],
        params["C1_highT"],
        params["C1_field_scale"],
    )
    return T_SP, delta, C1


def resolve_g_Q(params, delta):
    if params["g_Q_source"] == "delta_scaled":
        scale = float(delta) / max(float(delta_ref), 1e-15)
        return float(params["g_Q0"]) * scale
    if params["g_Q_source"] == "external":
        return float(params["g_Q_external"])
    return 0.0


def static_bright_dark_mixing(params, delta, C1):
    return (
        float(params["V0"])
        + float(params["lambda_delta"]) * float(delta)
        + float(params["lambda_C"]) * float(C1)
    )


def lowering_operator(n):
    op = np.zeros((n, n), dtype=complex)
    for upper in range(1, n):
        op[upper - 1, upper] = np.sqrt(upper)
    return op


def build_temperature_field_model(params):
    N_k = int(params["N_k"])
    if N_k != 1:
        raise NotImplementedError("Step 5 keeps N_k=1 to isolate trend scans.")
    k_array = np.array([0.0])
    k_weights = np.ones(1)

    n_bosons = int(params["n_bosons"])
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
    q_op = a + adag

    I_orb = np.eye(3, dtype=complex)
    I_ph = np.eye(n_bosons, dtype=complex)
    dim = 3 * n_bosons

    T_SP, delta, C1 = resolve_trends(params)
    V_BD_static = static_bright_dark_mixing(params, delta, C1)
    g_Q_eff = resolve_g_Q(params, delta)

    H_local = np.kron(H_orb, I_ph)
    H_phonon = float(params["omega_Q"]) * np.kron(I_orb, n_op)
    H_static_mix = V_BD_static * np.kron(L_bd, I_ph)
    H_dynamic_mix = g_Q_eff * np.kron(L_bd, q_op)

    H = H_local + H_phonon + H_static_mix + H_dynamic_mix
    mu = np.kron(mu_orb, I_ph)
    rho0 = np.zeros((dim, dim), dtype=complex)
    rho0[0, 0] = 1.0

    H_stack = H[None, :, :]
    mu_stack = mu[None, :, :]
    rho0_stack = rho0[None, :, :]

    c_ops = []
    if params["gamma_phonon"]:
        c_ops.append((np.kron(I_orb, a)[None, :, :], params["gamma_phonon"]))
    if params["gamma_orb"]:
        C_bg = np.zeros((3, 3), dtype=complex)
        C_bg[ket_g, ket_b] = 1.0
        C_dg = np.zeros((3, 3), dtype=complex)
        C_dg[ket_g, ket_d] = 1.0
        c_ops.append((np.kron(C_bg, I_ph)[None, :, :], params["gamma_orb"]))
        c_ops.append((np.kron(C_dg, I_ph)[None, :, :], params["gamma_orb"]))

    metadata = {
        "k_array": k_array,
        "k_weights": k_weights,
        "T_SP": float(T_SP),
        "delta": float(delta),
        "C1": float(C1),
        "delta_ref": float(delta_ref),
        "C1_highT_ref": float(C1_highT_ref),
        "V_BD_static": float(V_BD_static),
        "omega_Q": float(params["omega_Q"]),
        "g_Q_eff": float(g_Q_eff),
        "dim": dim,
    }
    return H_stack, mu_stack, c_ops, rho0_stack, metadata


# %% [markdown]
# ## Solver Setup

# %%
def make_solver(params):
    H, mu, c_ops, rho0, meta = build_temperature_field_model(params)

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
print("Scan family:", active_params["scan_family"])
print("Scan variable:", active_params["scan_variable"], active_params["scan_value"])
print("Hilbert dimension:", meta["dim"])
print("T:", active_params["T"])
print("B:", active_params["B"])
print("T_SP:", meta["T_SP"])
print("delta:", meta["delta"])
print("C1:", meta["C1"])
print("V_BD_static:", meta["V_BD_static"])
print("omega_Q:", meta["omega_Q"])
print("g_Q_eff:", meta["g_Q_eff"])
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

[(p.name, p.component, p.interactions, p.coherence_orders) for p in pathways]


# %% [markdown]
# ## Spectrum Calculation

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
save_pdf = False
plot_pathways = [p.name for p in rephasing_pathways]
plot_totals = ["rephasing"]
plot_view = "real"
plot_normalization = "individual"

case_label = (
    f"{active_params['label']}"
    f"_T_{safe_float_label(active_params['T'])}"
    f"_B_{safe_float_label(active_params['B'])}"
    f"_Eta_{safe_float_label(solver_params['Eta'])}"
    f"_Vstatic_{safe_float_label(meta['V_BD_static'])}"
    f"_delta_{safe_float_label(meta['delta'])}"
    f"_C1_{safe_float_label(meta['C1'])}"
    f"_gQ_{safe_float_label(meta['g_Q_eff'])}"
)

figure_directory = FIGURES_DIR / active_params["scan_family"] / active_params["label"]
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
        "test": "etape_5",
        "run_label": case_label,
        "active_case_key": active_case_key,
        "scan_family": active_params["scan_family"],
        "scan_variable": active_params["scan_variable"],
        "scan_value": float(active_params["scan_value"]),
        "channel": active_params["channel"],
        "model_parameters": active_params,
        "solver_parameters": solver_params,
        "derived": {
            "T": float(active_params["T"]),
            "B": float(active_params["B"]),
            "T_SP": float(meta["T_SP"]),
            "delta": float(meta["delta"]),
            "C1": float(meta["C1"]),
            "delta_ref": float(meta["delta_ref"]),
            "C1_highT_ref": float(meta["C1_highT_ref"]),
            "V_BD_static": float(meta["V_BD_static"]),
            "omega_Q": float(meta["omega_Q"]),
            "g_Q_eff": float(meta["g_Q_eff"]),
            "N_k": int(len(meta["k_array"])),
            "weight_sum": float(np.sum(meta["k_weights"])),
            "hilbert_dimension": int(meta["dim"]),
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
    lines.append("Test: etape_5")
    lines.append(f"Run label: {case_label}")
    lines.append(f"Active case: {active_case_key}")
    lines.append(f"Scan family: {active_params['scan_family']}")
    lines.append(f"Channel: {active_params['channel']}")
    lines.append("")
    lines.append("Temperature-field scan model")
    lines.append("----------------------------")
    lines.append("H_mix = V_BD_static L_BD + g_Q_eff Q L_BD")
    lines.append("V_BD_static = V0 + lambda_delta * delta(T,B) + lambda_C * C1(T,B)")
    lines.append(f"T: {active_params['T']}")
    lines.append(f"B: {active_params['B']}")
    lines.append(f"T_SP: {meta['T_SP']}")
    lines.append(f"delta: {meta['delta']}")
    lines.append(f"C1: {meta['C1']}")
    lines.append(f"V_BD_static: {meta['V_BD_static']}")
    lines.append(f"omega_Q: {meta['omega_Q']}")
    lines.append(f"g_Q_eff: {meta['g_Q_eff']}")
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
    lines.append(f"plot_pathways: {plot_pathways}")
    lines.append(f"plot_totals: {plot_totals}")
    lines.append(f"view: {plot_view}")
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
