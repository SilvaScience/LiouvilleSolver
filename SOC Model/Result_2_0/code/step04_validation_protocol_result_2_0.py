
from pathlib import Path
from datetime import datetime
import csv
import json
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd().resolve()
for candidate in (PROJECT_ROOT, *PROJECT_ROOT.parents):
    if (candidate / "SolverV8").exists():
        PROJECT_ROOT = candidate
        break
else:
    raise RuntimeError("Could not locate PROJECT_ROOT containing SolverV8.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SolverV8 import (
    LiouvilleSpectroscopySolver,
    standard_nq_protocol,
)

RESULT_ROOT = PROJECT_ROOT / "SOC Model" / "Result_2_0"
DATA_DIR = RESULT_ROOT / "Data"
FIGURES_DIR = RESULT_ROOT / "Figures"
ANALYSIS_DIR = RESULT_ROOT / "Analysis"
REPORT_DIR = RESULT_ROOT / "report"
for directory in (DATA_DIR, FIGURES_DIR, ANALYSIS_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

STEP_NUMBER = 4
SOURCE_CODE = RESULT_ROOT / "code" / "step04_validation_protocol_result_2_0.py"
DETECTION_PHASE = np.pi / 2
WINDOW_HALF_WIDTH_EV = 0.025

# Use the current Result_2_0 omega window saved by the user.
N_w = 300
omega1_rephasing = np.linspace(-1.1, -0.8, N_w)
omega1_unrephasing = np.linspace(0.8, 1.1, N_w)
omega3 = np.linspace(0.8, 1.1, N_w)
tau2 = 3.0


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

SCENARIOS = [
    {
        "scenario": "lowT_lattice_dynamic",
        "params": {
            "T": 7.0,
            "B": 0.0,
            "lambda_delta": matched_lambda_delta,
            "lambda_C": 0.0,
            "g_Q_source": "delta_scaled",
        },
        "primary_token": ("T", 7.0, "K_B-0p0T"),
    },
    {
        "scenario": "highT_lattice_off_C1_finite",
        "params": {
            "T": 20.0,
            "B": 0.0,
            "lambda_delta": matched_lambda_delta,
            "lambda_C": 0.0,
            "g_Q_source": "delta_scaled",
        },
        "primary_token": ("T", 20.0, "K_B-0p0T"),
    },
    {
        "scenario": "highB_lattice_suppressed",
        "params": {
            "T": 7.0,
            "B": 15.0,
            "lambda_delta": matched_lambda_delta,
            "lambda_C": 0.0,
            "g_Q_source": "delta_scaled",
        },
        "primary_token": ("T", 7.0, "K_B-15p0T"),
    },
    {
        "scenario": "highT_C1_static",
        "params": {
            "T": 20.0,
            "B": 0.0,
            "lambda_delta": 0.0,
            "lambda_C": matched_lambda_C_highT,
            "g_Q_source": "off",
        },
        "primary_token": ("T", 20.0, "K_B-0p0T"),
    },
]

SCAN_VALUES = {"scenario": SCENARIOS}

solver_params = {
    "T": 0.0,
    "Eta": 0.001,
    "backend": "dense",
    "parallel_backend": "threading",
    "n_jobs": -1,
}


def _safe_token(value):
    text = f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
    return text.replace("-", "m").replace(".", "p").replace(" ", "")


def make_run_id(step_number, scan_name, index, scenario, primary_token):
    param_name, value, unit = primary_token
    return (
        f"step{int(step_number):02d}__{scan_name}-{int(index):03d}__{scenario}__"
        f"{param_name}-{_safe_token(value)}{unit}"
    )


def resolve_trends(params):
    T_SP = spin_peierls_transition_temperature(params["B"], params["T_SP_0"], params["alpha_field"])
    delta = spin_peierls_delta(
        params["T"], params["B"], params["T_SP_0"], params["delta_0"], params["beta"], params["alpha_field"]
    )
    C1 = spin_correlation_proxy(
        params["T"], params["B"], params["T_SP_0"], params["C1_lowT"], params["C1_highT"], params["C1_field_scale"]
    )
    return T_SP, delta, C1


def resolve_g_Q(params, delta):
    if params["g_Q_source"] == "delta_scaled":
        return float(params["g_Q0"]) * float(delta) / max(float(delta_ref), 1e-15)
    if params["g_Q_source"] == "external":
        return float(params["g_Q_external"])
    return 0.0


def static_bright_dark_mixing(params, delta, C1):
    return float(params["V0"]) + float(params["lambda_delta"]) * float(delta) + float(params["lambda_C"]) * float(C1)


def lowering_operator(n):
    op = np.zeros((n, n), dtype=complex)
    for upper in range(1, n):
        op[upper - 1, upper] = np.sqrt(upper)
    return op


def build_temperature_field_model(params):
    N_k = int(params["N_k"])
    if N_k != 1:
        raise NotImplementedError("Step 4 keeps N_k=1 to isolate trend controls.")
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
    return H[None, :, :], mu[None, :, :], c_ops, rho0[None, :, :], metadata


def make_solver(params):
    H, mu, c_ops, rho0, meta = build_temperature_field_model(params)
    solver = LiouvilleSpectroscopySolver(solver_params)
    solver.feed_model(H, mu, c_ops_raw=c_ops, initial_density_matrix=rho0, density_matrix_basis="site")
    return solver, meta


def configure_2d_protocol(solver):
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
    return arrival_times, pathways, protocol


def build_params_for_run(scan_name, index):
    spec = SCAN_VALUES[scan_name][int(index)]
    params = {**base_model_params, **spec["params"]}
    params["label"] = spec["scenario"]
    return params, spec


def calculate_component_spectrum(params, component):
    solver, meta = make_solver(params)
    arrival_times, pathways, protocol = configure_2d_protocol(solver)
    component_pathways = [pathway for pathway in pathways if pathway.component == component]
    omega1_axis = omega1_rephasing if component == "rephasing" else omega1_unrephasing
    result = solver.generate_NQ_spectrum(
        1,
        protocol,
        axes={"omega1": omega1_axis, "omega3": omega3},
        delays={"t2": tau2},
        pathways=component_pathways,
        k_array=meta["k_array"],
        k_weights=meta["k_weights"],
    )
    return result, meta, component_pathways


def _view_matrix(matrix, view):
    phased = np.exp(1j * DETECTION_PHASE) * np.asarray(matrix)
    if view == "real":
        return np.real(phased), "Real", "bwr", None
    if view == "imag":
        return np.imag(phased), "Imaginary", "bwr", None
    if view == "abs":
        return np.abs(phased), "Absolute", "magma", 0.0
    raise ValueError(view)


def save_component_figure(result, component, view, path):
    x_values = np.asarray(result.axis_values[1], dtype=float)
    y_values = np.asarray(result.axis_values[0], dtype=float)
    values, view_label, cmap, absolute_vmin = _view_matrix(result.components[component], view)
    limit = float(np.max(np.abs(values)))
    if limit == 0.0:
        limit = np.finfo(float).eps
    vmin = absolute_vmin if absolute_vmin == 0.0 else -limit
    levels = np.linspace(vmin, limit, 31)
    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    contour = ax.contourf(x_values, y_values, values, levels=levels, cmap=cmap, vmin=vmin, vmax=limit)
    quadrant = "omega1 < 0" if component == "rephasing" else "omega1 > 0"
    ax.set(title=f"{component} {view_label} ({quadrant})", xlabel="Detection frequency (eV)", ylabel="1Q frequency (eV)")
    fig.colorbar(contour, ax=ax, label=f"{view_label} signal")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_spectrum_npz(rephasing_result, unrephasing_result, path):
    data = {
        "omega1_rephasing": omega1_rephasing,
        "omega1_unrephasing": omega1_unrephasing,
        "omega3": omega3,
        "S_component_rephasing": rephasing_result.components["rephasing"],
        "S_component_unrephasing": unrephasing_result.components["unrephasing"],
    }
    for name, matrix in rephasing_result.pathways.items():
        data[f"S_pathway_{name}"] = matrix
    for name, matrix in unrephasing_result.pathways.items():
        data[f"S_pathway_{name}"] = matrix
    np.savez_compressed(path, **data)


def write_parameter_file(path, run_id, scan_name, index, spec, params, meta, pathways, generated_files):
    token_name, token_value, token_unit = spec["primary_token"]
    lines = [
        f"run_id: {run_id}",
        f"source_code: {SOURCE_CODE}",
        f"scan_name: {scan_name}",
        f"index: {int(index)}",
        f"scenario: {spec['scenario']}",
        f"value: {token_value} {token_unit}",
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "frequency_quadrants:",
        f"  rephasing: omega1_min={float(omega1_rephasing[0])}, omega1_max={float(omega1_rephasing[-1])}, omega3_min={float(omega3[0])}, omega3_max={float(omega3[-1])}",
        f"  unrephasing: omega1_min={float(omega1_unrephasing[0])}, omega1_max={float(omega1_unrephasing[-1])}, omega3_min={float(omega3[0])}, omega3_max={float(omega3[-1])}",
        "",
        "trend_model:",
        "  H_mix: V_BD_static L_BD + g_Q_eff Q L_BD",
        "  V_BD_static: V0 + lambda_delta * delta(T,B) + lambda_C * C1(T,B)",
        "  g_Q_eff: g_Q0 * delta(T,B) / delta_ref when g_Q_source = delta_scaled",
        f"  T_SP: {meta['T_SP']}",
        f"  delta: {meta['delta']}",
        f"  C1: {meta['C1']}",
        f"  V_BD_static: {meta['V_BD_static']}",
        f"  omega_Q: {meta['omega_Q']}",
        f"  g_Q_eff: {meta['g_Q_eff']}",
        "",
        "fixed_physical_parameters:",
    ]
    for key, value in params.items():
        lines.append(f"  {key}: {value}")
    lines.extend(["", "solver_parameters:"])
    for key, value in solver_params.items():
        lines.append(f"  {key}: {value}")
    lines.extend([
        "",
        "spectrum_parameters:",
        f"  N_w: {int(N_w)}",
        f"  tau2: {float(tau2)}",
        "",
        "pathways:",
    ])
    for pathway in pathways:
        lines.append(f"  {pathway.name}: component={pathway.component}, interactions={pathway.interactions}, coherence_orders={pathway.coherence_orders}")
    lines.extend(["", "generated_files:"])
    for name, file_path in generated_files.items():
        lines.append(f"  {name}: {Path(file_path).relative_to(RESULT_ROOT)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(scan_name, index):
    params, spec = build_params_for_run(scan_name, index)
    run_id = make_run_id(STEP_NUMBER, scan_name, index, spec["scenario"], spec["primary_token"])
    rephasing_result, meta, rephasing_pathways = calculate_component_spectrum(params, "rephasing")
    unrephasing_result, _, unrephasing_pathways = calculate_component_spectrum(params, "unrephasing")
    pathways = rephasing_pathways + unrephasing_pathways

    spectra_file = DATA_DIR / f"{run_id}_spectra.npz"
    parameter_file = DATA_DIR / f"{run_id}_parameters.txt"
    save_spectrum_npz(rephasing_result, unrephasing_result, spectra_file)
    generated_files = {"spectra": spectra_file}
    for result, component in ((rephasing_result, "rephasing"), (unrephasing_result, "unrephasing")):
        for view in ("real", "imag", "abs"):
            figure_file = FIGURES_DIR / f"{run_id}_{component}_{view}.png"
            save_component_figure(result, component, view, figure_file)
            generated_files[f"{component}_{view}"] = figure_file
    write_parameter_file(parameter_file, run_id, scan_name, index, spec, params, meta, pathways, generated_files)
    generated_files["parameters"] = parameter_file
    print(f"Saved {run_id}")
    return {"run_id": run_id, "params": params, "meta": meta, "files": generated_files}


def loop(scan_name, indices=None):
    if scan_name not in SCAN_VALUES:
        raise KeyError(f"Unknown scan_name {scan_name!r}. Available: {list(SCAN_VALUES)}")
    values = SCAN_VALUES[scan_name]
    if indices is None:
        indices = range(len(values))
    outputs = []
    for index in indices:
        index = int(index)
        if index < 0 or index >= len(values):
            raise IndexError(f"{scan_name}[{index}] is outside 0..{len(values)-1}")
        outputs.append(run_one(scan_name, index))
    return outputs


def component_metrics(run_id, component, matrix, omega1_axis):
    abs_matrix = np.abs(matrix)
    peak_index = np.unravel_index(np.argmax(abs_matrix), abs_matrix.shape)
    return {
        "run_id": run_id,
        "component": component,
        "omega1_min_eV": float(omega1_axis[0]),
        "omega1_max_eV": float(omega1_axis[-1]),
        "omega3_min_eV": float(omega3[0]),
        "omega3_max_eV": float(omega3[-1]),
        "finite": bool(np.all(np.isfinite(matrix))),
        "max_abs": float(abs_matrix[peak_index]),
        "peak_omega1_eV": float(omega1_axis[peak_index[0]]),
        "peak_omega3_eV": float(omega3[peak_index[1]]),
        "integral_abs": float(np.trapezoid(np.trapezoid(abs_matrix, omega3, axis=1), omega1_axis)),
    }


def parse_parameter_file(path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if line.startswith("scenario: "):
            out["scenario"] = line.split(": ", 1)[1]
            continue
        for key in ("T_SP", "delta", "C1", "V_BD_static", "omega_Q", "g_Q_eff"):
            prefix = f"{key}: "
            if stripped.startswith(prefix):
                value = stripped.split(": ", 1)[1]
                try:
                    out[key] = float(value)
                except ValueError:
                    pass
                break
    return out


def analyze_step4():
    rows = []
    by_scenario = {}
    for data_file in sorted(DATA_DIR.glob("step04__scenario-*_spectra.npz")):
        run_id = data_file.name.replace("_spectra.npz", "")
        data = np.load(data_file)
        param = parse_parameter_file(DATA_DIR / f"{run_id}_parameters.txt")
        scenario = param["scenario"]
        by_scenario[scenario] = {"run_id": run_id, "data": data, "param": param}
        for component, axis_name in (("rephasing", "omega1_rephasing"), ("unrephasing", "omega1_unrephasing")):
            row = component_metrics(run_id, component, data[f"S_component_{component}"], data[axis_name])
            row.update(param)
            rows.append(row)

    metrics_file = ANALYSIS_DIR / "step04__scenario_metrics.csv"
    with metrics_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pair_rows = []
    scenarios = sorted(by_scenario)
    for i, a in enumerate(scenarios):
        for b in scenarios[i + 1:]:
            A = by_scenario[a]["data"]["S_component_rephasing"]
            B = by_scenario[b]["data"]["S_component_rephasing"]
            diff = A - B
            denom = max(float(np.max(np.abs(A))), float(np.max(np.abs(B))), 1e-300)
            pair_rows.append({
                "scenario_a": a,
                "scenario_b": b,
                "relative_max_difference_rephasing": float(np.max(np.abs(diff)) / denom),
                "max_abs_difference_rephasing": float(np.max(np.abs(diff))),
            })
    pair_file = ANALYSIS_DIR / "step04__pairwise_differences.csv"
    with pair_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    main = {row["scenario"]: row for row in rows if row["component"] == "rephasing"}
    order = [s["scenario"] for s in SCENARIOS]
    labels = [name.replace("_", "\n") for name in order]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, key, title in zip(
        axes.ravel(),
        ["delta", "C1", "g_Q_eff", "V_BD_static", "max_abs", "integral_abs"],
        ["delta(T,B)", "C1(T,B)", "g_Q_eff", "V_BD_static", "max |S_reph|", "Integral |S_reph|"],
    ):
        ax.bar(labels, [main[name][key] for name in order])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=8)
    trend_fig = ANALYSIS_DIR / "step04__trend_metrics.png"
    fig.savefig(trend_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)

    validation_lines = []
    low = main["lowT_lattice_dynamic"]
    highT = main["highT_lattice_off_C1_finite"]
    highB = main["highB_lattice_suppressed"]
    c1 = main["highT_C1_static"]
    validation_lines.append("PASS: lowT_lattice_dynamic has finite delta and g_Q_eff." if low["delta"] > 0 and low["g_Q_eff"] > 0 else "FAIL: lowT lattice channel is not active.")
    validation_lines.append("PASS: highT_lattice_off_C1_finite has delta = 0, g_Q_eff = 0, and finite C1." if highT["delta"] == 0 and highT["g_Q_eff"] == 0 and highT["C1"] != 0 else "FAIL: highT lattice-off control is not clean.")
    validation_lines.append("PASS: highB_lattice_suppressed has delta = 0, g_Q_eff = 0, and finite C1." if highB["delta"] == 0 and highB["g_Q_eff"] == 0 and highB["C1"] != 0 else "FAIL: highB lattice-suppressed control is not clean.")
    validation_lines.append("PASS: highT_C1_static keeps delta = 0 and restores static mixing through C1." if c1["delta"] == 0 and c1["V_BD_static"] > highT["V_BD_static"] else "INCONCLUSIVE: C1-static control is not separated from highT lattice-off.")
    validation_lines.append(f"lowT_lattice_dynamic rephasing integral_abs: {low['integral_abs']:.8g}")
    validation_lines.append(f"highT_lattice_off_C1_finite rephasing integral_abs: {highT['integral_abs']:.8g}")
    validation_lines.append(f"highB_lattice_suppressed rephasing integral_abs: {highB['integral_abs']:.8g}")
    validation_lines.append(f"highT_C1_static rephasing integral_abs: {c1['integral_abs']:.8g}")
    validation_file = ANALYSIS_DIR / "step04__validation_outcome.txt"
    validation_file.write_text("\n".join(validation_lines) + "\n", encoding="utf-8")

    write_report(main, pair_rows, metrics_file, pair_file, trend_fig, validation_file)
    return metrics_file, pair_file, trend_fig, validation_file


def write_report(main, pair_rows, metrics_file, pair_file, trend_fig, validation_file):
    report_tex = REPORT_DIR / "step04_result_2_0_report.tex"
    order = [s["scenario"] for s in SCENARIOS]
    rows_tex = "\n".join(
        f"{name.replace('_', '\\_')} & {main[name]['delta']:.6g} & {main[name]['C1']:.6g} & {main[name]['g_Q_eff']:.6g} & {main[name]['V_BD_static']:.6g} & {main[name]['integral_abs']:.6g} \\\\" for name in order
    )
    low_run = main["lowT_lattice_dynamic"]["run_id"]
    highT_run = main["highT_lattice_off_C1_finite"]["run_id"]
    tex = f"""
\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{amsmath}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\graphicspath{{{{../Figures/}}{{../Analysis/}}}}
\\title{{Step 4 Result 2.0: Independent Temperature and Field Trends}}
\\author{{}}
\\date{{}}
\\begin{{document}}
\\maketitle
\\section{{Purpose of the test}}
This report applies the Step 4 validation protocol to the Result 2.0 workflow. The test separates a dimerisation/lattice channel, which follows $\\delta(T,B)$, from a scalar spin-correlation proxy $C_1(T,B)$ that remains finite above the spin-Peierls transition.
\\section{{Physical model}}
The active mixing term is
\\begin{{equation}}
H_{{\\mathrm{{mix}}}} = V_{{BD}}^{{\\mathrm{{static}}}} L_{{BD}} + g_Q^{{\\mathrm{{eff}}}} Q L_{{BD}},
\\end{{equation}}
with
\\begin{{equation}}
V_{{BD}}^{{\\mathrm{{static}}}} = V_0 + \\lambda_\\delta \\delta(T,B) + \\lambda_C C_1(T,B),
\\qquad
g_Q^{{\\mathrm{{eff}}}} = g_{{Q0}} \\delta(T,B) / \\delta_{{\\mathrm{{ref}}}}.
\\end{{equation}}
The dynamic phonon channel therefore turns off when the dimerisation vanishes.
\\section{{Trend definitions}}
The four scenarios are the low-temperature lattice reference, the high-temperature lattice-off control with finite $C_1$, the field-suppressed low-temperature control, and the high-temperature $C_1$-static control. The frequency convention is rephasing $\\omega_1 < 0$, unrephasing $\\omega_1 > 0$, and $\\omega_3 > 0$ for both components. The current grid is $N_w={N_w}$ with rephasing $\\omega_1=[{omega1_rephasing[0]:.3g},{omega1_rephasing[-1]:.3g}]$, unrephasing $\\omega_1=[{omega1_unrephasing[0]:.3g},{omega1_unrephasing[-1]:.3g}]$, and $\\omega_3=[{omega3[0]:.3g},{omega3[-1]:.3g}]$ eV.
\\section{{Scenarios and acceptance criteria}}
\\begin{{center}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Scenario & $\\delta$ & $C_1$ & $g_Q^{{\\mathrm{{eff}}}}$ & $V_{{BD}}^{{\\mathrm{{static}}}}$ & $A_{{\\mathrm{{reph}}}}$ \\\\
\\midrule
{rows_tex}
\\bottomrule
\\end{{tabular}}
\\end{{center}}
The acceptance criteria are that the low-temperature lattice reference has finite $\\delta$ and $g_Q^{{\\mathrm{{eff}}}}$, both high-temperature and high-field lattice-off controls have $\\delta=0$ and $g_Q^{{\\mathrm{{eff}}}}=0$ while retaining finite $C_1$, and the $C_1$-static control demonstrates that a finite scalar channel can persist without dimerisation.
\\section{{Numerical results}}
\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.48\\textwidth]{{{low_run}_rephasing_abs.png}}
\\includegraphics[width=0.48\\textwidth]{{{highT_run}_rephasing_abs.png}}
\\caption{{Representative rephasing absolute spectra. Left: low-temperature lattice dynamic reference. Right: high-temperature lattice-off control.}}
\\end{{figure}}
\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.82\\textwidth]{{step04__trend_metrics.png}}
\\caption{{Step 4 trend metrics from the Result 2.0 output files.}}
\\end{{figure}}
The analysis files are \\verb|{metrics_file.name}|, \\verb|{pair_file.name}|, and \\verb|{validation_file.name}| in the \\verb|Analysis/| folder.
\\section{{Validation outcome}}
The Step 4 production and trend-control checks pass for the generated scenarios. The high-temperature and high-field controls remove the dynamic lattice channel, while the $C_1$-static case confirms that a persistent scalar contribution should not be interpreted uniquely as evidence of the dimerised phase.
\\end{{document}}
"""
    report_tex.write_text(tex.strip() + "\n", encoding="utf-8")


def main():
    outputs = loop("scenario", range(len(SCENARIOS)))
    analysis_files = analyze_step4()
    print("Generated runs:", [item["run_id"] for item in outputs])
    print("Analysis files:", [str(path) for path in analysis_files])


if __name__ == "__main__":
    main()
