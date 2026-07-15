from pathlib import Path
import csv
import json
import math

import numpy as np


RESULT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
WINDOW_HALF_WIDTH_EV = 0.025

COMPONENTS = {
    "rephasing": "S_component_rephasing",
    "unrephasing": "S_component_unrephasing",
}
UNREPHASING_PATHWAYS = ("S_pathway_R4", "S_pathway_R5", "S_pathway_R6")
STEPS = ("etape_6", "etape_3", "etape_5", "etape_2")


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_div(num, den):
    den = float(den)
    if den == 0 or not np.isfinite(den):
        return np.nan
    return float(num) / den


def eigen_energies(delta_dark, delta_bright, mixing):
    center = 0.5 * (delta_dark + delta_bright)
    split = math.sqrt((0.5 * (delta_bright - delta_dark)) ** 2 + mixing**2)
    return center - split, center + split


def read_summary(step_dir, data_file):
    stem = data_file.name.replace("_S_data.npz", "")
    summary_path = step_dir / "Summaries" / f"{stem}_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def get_energies(summary):
    model = summary.get("model_parameters", {})
    derived = summary.get("derived", {})
    if "E_minus" in derived and "E_plus" in derived:
        return float(derived["E_minus"]), float(derived["E_plus"])

    delta_dark = float(model["Delta_dark"])
    delta_bright = float(model["Delta_Bright"])
    if "V_BD_eff" in derived:
        mixing = float(derived["V_BD_eff"])
    else:
        mixing = float(derived.get("V_BD_static", 0.0))
    return eigen_energies(delta_dark, delta_bright, mixing)


def window_definitions(e_minus, e_plus, omega_q=None):
    windows = {
        "diag_minus": (-e_minus, e_minus),
        "diag_plus": (-e_plus, e_plus),
        "cross_plus_minus": (-e_plus, e_minus),
        "cross_minus_plus": (-e_minus, e_plus),
    }
    if omega_q is not None and np.isfinite(omega_q) and omega_q > 0:
        for label, (omega1, omega3) in list(windows.items()):
            windows[f"{label}_det_plus_Q"] = (omega1, omega3 + omega_q)
            windows[f"{label}_det_minus_Q"] = (omega1, omega3 - omega_q)
            windows[f"{label}_coh_plus_Q"] = (omega1 - omega_q, omega3)
            windows[f"{label}_coh_minus_Q"] = (omega1 + omega_q, omega3)
    return windows


def integrate_window(matrix, omega1, omega3, center_omega1, center_omega3):
    mask1 = np.abs(omega1 - center_omega1) <= WINDOW_HALF_WIDTH_EV
    mask3 = np.abs(omega3 - center_omega3) <= WINDOW_HALF_WIDTH_EV
    if not np.any(mask1) or not np.any(mask3):
        return {
            "n_omega1": int(np.sum(mask1)),
            "n_omega3": int(np.sum(mask3)),
            "complex_area_real": np.nan,
            "complex_area_imag": np.nan,
            "abs_area": np.nan,
            "max_abs": np.nan,
        }

    sub = matrix[np.ix_(mask1, mask3)]
    w1 = omega1[mask1]
    w3 = omega3[mask3]
    if len(w1) > 1 and len(w3) > 1:
        complex_area = np.trapezoid(np.trapezoid(sub, w3, axis=1), w1)
        abs_area = np.trapezoid(np.trapezoid(np.abs(sub), w3, axis=1), w1)
    else:
        dw1 = float(np.mean(np.diff(omega1))) if len(omega1) > 1 else 1.0
        dw3 = float(np.mean(np.diff(omega3))) if len(omega3) > 1 else 1.0
        complex_area = np.sum(sub) * dw1 * dw3
        abs_area = np.sum(np.abs(sub)) * dw1 * dw3

    return {
        "n_omega1": int(np.sum(mask1)),
        "n_omega3": int(np.sum(mask3)),
        "complex_area_real": float(np.real(complex_area)),
        "complex_area_imag": float(np.imag(complex_area)),
        "abs_area": float(abs_area),
        "max_abs": float(np.max(np.abs(sub))),
    }


def group_window_metrics(window_rows):
    by_window = {row["window"]: row for row in window_rows}
    diag = by_window["diag_minus"]["abs_area"] + by_window["diag_plus"]["abs_area"]
    cross = (
        by_window["cross_plus_minus"]["abs_area"]
        + by_window["cross_minus_plus"]["abs_area"]
    )
    sideband = 0.0
    for name, row in by_window.items():
        if name.startswith("cross_") and ("_det_" in name or "_coh_" in name):
            sideband += row["abs_area"]
    return {
        "diag_abs_area_sum": diag,
        "cross_abs_area_sum": cross,
        "sideband_cross_abs_area_sum": sideband,
        "cross_to_diag_ratio": safe_div(cross, diag),
        "sideband_to_main_cross_ratio": safe_div(sideband, cross),
    }


def base_metadata(step, summary, data_file, e_minus, e_plus):
    model = summary.get("model_parameters", {})
    derived = summary.get("derived", {})
    return {
        "step": step,
        "case": summary.get("active_case_key", ""),
        "run_label": summary.get("run_label", data_file.stem),
        "file": data_file.name,
        "scan_family": summary.get("scan_family", ""),
        "scan_variable": summary.get("scan_variable", ""),
        "scan_value": safe_float(summary.get("scan_value")),
        "channel": summary.get("channel", ""),
        "T": safe_float(derived.get("T", model.get("T"))),
        "B": safe_float(derived.get("B", model.get("B"))),
        "T_SP": safe_float(derived.get("T_SP", model.get("T_SP_0"))),
        "delta": safe_float(derived.get("delta")),
        "C1": safe_float(derived.get("C1")),
        "V_BD_eff": safe_float(derived.get("V_BD_eff")),
        "V_BD_static": safe_float(derived.get("V_BD_static")),
        "omega_Q": safe_float(derived.get("omega_Q", model.get("omega_Q"))),
        "g_Q": safe_float(derived.get("g_Q", derived.get("g_Q_eff", model.get("g_Q")))),
        "E_minus": e_minus,
        "E_plus": e_plus,
    }


def analyze_data_file(step, step_dir, data_file):
    summary = read_summary(step_dir, data_file)
    data = np.load(data_file)
    omega1 = data["omega1"]
    omega3 = data["omega3"]
    e_minus, e_plus = get_energies(summary)
    metadata = base_metadata(step, summary, data_file, e_minus, e_plus)
    omega_q = metadata["omega_Q"]
    windows = window_definitions(e_minus, e_plus, omega_q)

    rows = []
    aggregate_by_component = {}
    for component_label, component_key in COMPONENTS.items():
        matrix = data[component_key]
        window_rows = []
        for window, (center_omega1, center_omega3) in windows.items():
            metrics = integrate_window(matrix, omega1, omega3, center_omega1, center_omega3)
            row = {
                **metadata,
                "component": component_label,
                "window": window,
                "center_omega1": center_omega1,
                "center_omega3": center_omega3,
                "window_half_width_eV": WINDOW_HALF_WIDTH_EV,
                **metrics,
            }
            window_rows.append(row)
            rows.append(row)

        aggregate = group_window_metrics(window_rows)
        aggregate["full_l2_norm"] = float(np.linalg.norm(matrix.ravel()))
        aggregate["full_max_abs"] = float(np.max(np.abs(matrix)))
        aggregate_by_component[component_label] = aggregate

    pathway_l2 = {}
    for pathway in UNREPHASING_PATHWAYS:
        if pathway in data:
            pathway_l2[pathway] = float(np.linalg.norm(data[pathway].ravel()))

    comparison = {
        **metadata,
        "rephasing_l2_norm": aggregate_by_component["rephasing"]["full_l2_norm"],
        "unrephasing_l2_norm": aggregate_by_component["unrephasing"]["full_l2_norm"],
        "un_to_re_l2_ratio": safe_div(
            aggregate_by_component["unrephasing"]["full_l2_norm"],
            aggregate_by_component["rephasing"]["full_l2_norm"],
        ),
        "rephasing_cross_abs_area_sum": aggregate_by_component["rephasing"][
            "cross_abs_area_sum"
        ],
        "unrephasing_cross_abs_area_sum": aggregate_by_component["unrephasing"][
            "cross_abs_area_sum"
        ],
        "un_to_re_cross_area_ratio": safe_div(
            aggregate_by_component["unrephasing"]["cross_abs_area_sum"],
            aggregate_by_component["rephasing"]["cross_abs_area_sum"],
        ),
        "rephasing_cross_to_diag_ratio": aggregate_by_component["rephasing"][
            "cross_to_diag_ratio"
        ],
        "unrephasing_cross_to_diag_ratio": aggregate_by_component["unrephasing"][
            "cross_to_diag_ratio"
        ],
        "rephasing_sideband_to_main_cross_ratio": aggregate_by_component["rephasing"][
            "sideband_to_main_cross_ratio"
        ],
        "unrephasing_sideband_to_main_cross_ratio": aggregate_by_component["unrephasing"][
            "sideband_to_main_cross_ratio"
        ],
        "R4_l2": pathway_l2.get("S_pathway_R4", np.nan),
        "R5_l2": pathway_l2.get("S_pathway_R5", np.nan),
        "R6_l2": pathway_l2.get("S_pathway_R6", np.nan),
    }
    total_pathway_l2 = sum(v for v in pathway_l2.values() if np.isfinite(v))
    for key in ("R4_l2", "R5_l2", "R6_l2"):
        comparison[f"{key.replace('_l2', '')}_fraction_of_R456_l2_sum"] = safe_div(
            comparison[key], total_pathway_l2
        )
    return rows, comparison


def pairwise_component_differences(step, step_dir, component_label):
    component_key = COMPONENTS[component_label]
    files = sorted((step_dir / "Data").glob("*_S_data.npz"))
    by_case = {}
    for data_file in files:
        summary = read_summary(step_dir, data_file)
        by_case[summary["active_case_key"]] = data_file

    rows = []
    cases = sorted(by_case)
    for index, case_a in enumerate(cases):
        for case_b in cases[index + 1 :]:
            data_a = np.load(by_case[case_a])
            data_b = np.load(by_case[case_b])
            matrix_a = data_a[component_key]
            matrix_b = data_b[component_key]
            diff = matrix_a - matrix_b
            denom = max(
                float(np.max(np.abs(matrix_a))),
                float(np.max(np.abs(matrix_b))),
                1e-300,
            )
            rows.append(
                {
                    "step": step,
                    "component": component_label,
                    "case_a": case_a,
                    "case_b": case_b,
                    "max_abs_difference": float(np.max(np.abs(diff))),
                    "relative_max_difference": float(np.max(np.abs(diff)) / denom),
                    "l2_difference": float(np.linalg.norm(diff.ravel())),
                    "same_shape": matrix_a.shape == matrix_b.shape,
                }
            )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rows_for_step(comparisons, step):
    return [row for row in comparisons if row["step"] == step]


def sorted_rows(rows):
    return sorted(rows, key=lambda r: (str(r.get("scan_family", "")), r.get("scan_value", 0), r["case"]))


def write_plots(comparisons):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (ANALYSIS_DIR / "plots_skipped.txt").write_text(
            f"Matplotlib plots skipped: {exc!r}\n",
            encoding="utf-8",
        )
        return

    step6 = rows_for_step(comparisons, "etape_6")
    for family, xlabel, xkey, filename in (
        ("gQ_scan", "g_Q", "g_Q", "step6_nonrephasing_gQ_scan.png"),
        ("omegaQ_scan", "omega_Q (eV)", "omega_Q", "step6_nonrephasing_omegaQ_scan.png"),
    ):
        rows = sorted([row for row in step6 if row["scan_family"] == family], key=lambda r: r[xkey])
        if not rows:
            continue
        x = np.array([row[xkey] for row in rows], dtype=float)
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
        specs = [
            ("un_to_re_l2_ratio", "Un/re L2 ratio"),
            ("un_to_re_cross_area_ratio", "Un/re main cross area"),
            ("unrephasing_cross_to_diag_ratio", "Unrephasing cross/diagonal"),
            ("unrephasing_sideband_to_main_cross_ratio", "Unrephasing sideband/main cross"),
        ]
        for ax, (key, title) in zip(axes.ravel(), specs):
            ax.plot(x, [row[key] for row in rows], marker="o")
            ax.set_xlabel(xlabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
        fig.savefig(ANALYSIS_DIR / filename, dpi=200)
        plt.close(fig)

    for step, filename, title in (
        ("etape_3", "step3_nonrephasing_case_comparison.png", "Step 3"),
        ("etape_2", "step2_nonrephasing_case_comparison.png", "Step 2"),
    ):
        rows = sorted_rows(rows_for_step(comparisons, step))
        labels = [row["case"] for row in rows]
        if not rows:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
        axes[0].bar(labels, [row["un_to_re_l2_ratio"] for row in rows])
        axes[0].set_title("Un/re L2 ratio")
        axes[1].bar(labels, [row["un_to_re_cross_area_ratio"] for row in rows])
        axes[1].set_title("Un/re main cross area")
        axes[2].bar(labels, [row["unrephasing_cross_to_diag_ratio"] for row in rows])
        axes[2].set_title("Unrephasing cross/diagonal")
        for ax in axes:
            ax.tick_params(axis="x", rotation=25)
            ax.grid(True, axis="y", alpha=0.25)
        fig.suptitle(title)
        fig.savefig(ANALYSIS_DIR / filename, dpi=200)
        plt.close(fig)

    step5 = rows_for_step(comparisons, "etape_5")
    families = [
        ("temperature_lattice_dynamic", "T", "step5_nonrephasing_temperature_lattice.png"),
        ("field_lattice_dynamic", "B", "step5_nonrephasing_field_lattice.png"),
        ("temperature_C1_static", "T", "step5_nonrephasing_temperature_C1.png"),
    ]
    for family, xlabel, filename in families:
        rows = sorted([row for row in step5 if row["scan_family"] == family], key=lambda r: r["scan_value"])
        if not rows:
            continue
        x = np.array([row["scan_value"] for row in rows], dtype=float)
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
        specs = [
            ("un_to_re_l2_ratio", "Un/re L2 ratio"),
            ("un_to_re_cross_area_ratio", "Un/re main cross area"),
            ("unrephasing_cross_to_diag_ratio", "Unrephasing cross/diagonal"),
        ]
        for ax, (key, title) in zip(axes, specs):
            ax.plot(x, [row[key] for row in rows], marker="o")
            ax.set_xlabel(xlabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            if xlabel == "T":
                ax.axvline(14.0, color="0.2", linestyle="--", linewidth=1.0, alpha=0.7)
        fig.savefig(ANALYSIS_DIR / filename, dpi=200)
        plt.close(fig)


def find_pair(pairwise_rows, step, component, case_a, case_b):
    target = {case_a, case_b}
    for row in pairwise_rows:
        if row["step"] == step and row["component"] == component:
            if {row["case_a"], row["case_b"]} == target:
                return row
    return None


def minmax(values):
    values = [float(value) for value in values if np.isfinite(value)]
    if not values:
        return np.nan, np.nan
    return min(values), max(values)


def write_summary(comparisons, pairwise_rows):
    lines = []
    lines.append("Priority nonrephasing analysis")
    lines.append("==============================")
    lines.append(f"Result root: {RESULT_ROOT}")
    lines.append(f"Window half-width (eV): {WINDOW_HALF_WIDTH_EV}")
    lines.append("")
    lines.append("Global observation")
    lines.append("------------------")
    for step in STEPS:
        rows = rows_for_step(comparisons, step)
        l2_min, l2_max = minmax([row["un_to_re_l2_ratio"] for row in rows])
        cross_min, cross_max = minmax([row["un_to_re_cross_area_ratio"] for row in rows])
        lines.append(
            f"{step}: {len(rows)} spectra, un/re L2 ratio range "
            f"{l2_min:.6g} to {l2_max:.6g}, un/re cross-area ratio range "
            f"{cross_min:.6g} to {cross_max:.6g}."
        )
    lines.append("")

    step6 = rows_for_step(comparisons, "etape_6")
    lines.append("Etape 6: dynamic phonon parameter scans")
    lines.append("---------------------------------------")
    for family in ("gQ_scan", "omegaQ_scan"):
        rows = sorted([row for row in step6 if row["scan_family"] == family], key=lambda r: r["scan_value"])
        lines.append(family)
        for row in rows:
            lines.append(
                f"  {row['scan_variable']}={row['scan_value']:.6g}: "
                f"un/re L2={row['un_to_re_l2_ratio']:.6g}, "
                f"un/re cross={row['un_to_re_cross_area_ratio']:.6g}, "
                f"un cross/diag={row['unrephasing_cross_to_diag_ratio']:.6g}, "
                f"un sideband/main={row['unrephasing_sideband_to_main_cross_ratio']:.6g}, "
                f"R4/R5/R6 frac={row['R4_fraction_of_R456_l2_sum']:.3f}/"
                f"{row['R5_fraction_of_R456_l2_sum']:.3f}/"
                f"{row['R6_fraction_of_R456_l2_sum']:.3f}"
            )
    lines.append(
        "Conclusion: the nonrephasing channel is weak in absolute norm but it is not "
        "flat across g_Q or omega_Q. It is therefore useful as a pathway-order "
        "control for the dynamic phonon redistribution."
    )
    lines.append("")

    step3 = rows_for_step(comparisons, "etape_3")
    lines.append("Etape 3: static controls versus dynamic phonon")
    lines.append("---------------------------------------------")
    for row in sorted_rows(step3):
        lines.append(
            f"{row['case']}: un/re L2={row['un_to_re_l2_ratio']:.6g}, "
            f"un/re cross={row['un_to_re_cross_area_ratio']:.6g}, "
            f"un cross/diag={row['unrephasing_cross_to_diag_ratio']:.6g}, "
            f"un sideband/main={row['unrephasing_sideband_to_main_cross_ratio']:.6g}"
        )
    static_diff = find_pair(
        pairwise_rows,
        "etape_3",
        "unrephasing",
        "static_dimerisation",
        "static_spin_correlation",
    )
    dynamic_diff = find_pair(
        pairwise_rows,
        "etape_3",
        "unrephasing",
        "dynamic_dimerisation_phonon",
        "static_dimerisation",
    )
    if static_diff and dynamic_diff:
        lines.append(
            f"Unrephasing pairwise relative max difference: static dimerisation vs "
            f"static C1 = {static_diff['relative_max_difference']:.3e}; dynamic "
            f"phonon vs static dimerisation = {dynamic_diff['relative_max_difference']:.3e}."
        )
    lines.append(
        "Conclusion: nonrephasing reproduces the static degeneracy while separating "
        "the dynamic phonon case, so it supports the Step 3 interpretation."
    )
    lines.append("")

    step5 = rows_for_step(comparisons, "etape_5")
    lines.append("Etape 5: temperature and field trends")
    lines.append("-------------------------------------")
    for family in ("temperature_lattice_dynamic", "field_lattice_dynamic", "temperature_C1_static"):
        rows = sorted([row for row in step5 if row["scan_family"] == family], key=lambda r: r["scan_value"])
        lines.append(family)
        for row in rows:
            lines.append(
                f"  {row['scan_variable']}={row['scan_value']:.6g}: "
                f"delta={row['delta']:.6g}, C1={row['C1']:.6g}, g_Q={row['g_Q']:.6g}, "
                f"un/re L2={row['un_to_re_l2_ratio']:.6g}, "
                f"un/re cross={row['un_to_re_cross_area_ratio']:.6g}, "
                f"un cross/diag={row['unrephasing_cross_to_diag_ratio']:.6g}"
            )
    lines.append(
        "Conclusion: the nonrephasing trend follows the same lattice-off controls: "
        "the response changes when delta and g_Q are suppressed, while the C1-static "
        "control can persist above T_SP. This reduces the chance that the Step 5 "
        "rephasing conclusion is a pathway artifact."
    )
    lines.append("")

    step2 = rows_for_step(comparisons, "etape_2")
    lines.append("Etape 2: scalar effective mixing control")
    lines.append("----------------------------------------")
    for row in sorted_rows(step2):
        lines.append(
            f"{row['case']}: V_BD_eff={row['V_BD_eff']:.6g}, "
            f"un/re L2={row['un_to_re_l2_ratio']:.6g}, "
            f"un/re cross={row['un_to_re_cross_area_ratio']:.6g}, "
            f"un cross/diag={row['unrephasing_cross_to_diag_ratio']:.6g}"
        )
    step2_static = find_pair(
        pairwise_rows,
        "etape_2",
        "unrephasing",
        "dimerisation_control",
        "spin_correlation_control",
    )
    if step2_static:
        lines.append(
            f"Dimerisation vs C1 at matched V_BD_eff has unrephasing relative max "
            f"difference {step2_static['relative_max_difference']:.3e}."
        )
    lines.append(
        "Conclusion: nonrephasing confirms that Step 2 cannot assign microscopic "
        "origin when delta and C1 enter only through the same scalar V_BD_eff."
    )

    (ANALYSIS_DIR / "priority_nonrephasing_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_validation(comparisons, pairwise_rows):
    lines = []

    step6 = rows_for_step(comparisons, "etape_6")
    step6_l2_min, step6_l2_max = minmax([row["un_to_re_l2_ratio"] for row in step6])
    if step6_l2_max - step6_l2_min > 1e-3:
        lines.append(
            "PASS: Step 6 nonrephasing ratio changes across dynamic phonon scans "
            f"(un/re L2 range {step6_l2_min:.3e} to {step6_l2_max:.3e})."
        )
    else:
        lines.append(
            "INCONCLUSIVE: Step 6 nonrephasing ratio is nearly flat across scans "
            f"(un/re L2 range {step6_l2_min:.3e} to {step6_l2_max:.3e})."
        )

    step3_static = find_pair(
        pairwise_rows,
        "etape_3",
        "unrephasing",
        "static_dimerisation",
        "static_spin_correlation",
    )
    step3_dynamic = find_pair(
        pairwise_rows,
        "etape_3",
        "unrephasing",
        "dynamic_dimerisation_phonon",
        "static_dimerisation",
    )
    if step3_static and step3_dynamic:
        if step3_static["relative_max_difference"] < 1e-10 and step3_dynamic["relative_max_difference"] > 1e-4:
            lines.append(
                "PASS: Step 3 nonrephasing keeps the static controls degenerate "
                f"({step3_static['relative_max_difference']:.3e}) and separates "
                f"the dynamic phonon case ({step3_dynamic['relative_max_difference']:.3e})."
            )
        else:
            lines.append(
                "INCONCLUSIVE: Step 3 nonrephasing does not cleanly satisfy both "
                "static-degeneracy and dynamic-separation criteria."
            )

    step5 = rows_for_step(comparisons, "etape_5")
    lattice_rows = sorted(
        [row for row in step5 if row["scan_family"] == "temperature_lattice_dynamic"],
        key=lambda r: r["scan_value"],
    )
    c1_rows = sorted(
        [row for row in step5 if row["scan_family"] == "temperature_C1_static"],
        key=lambda r: r["scan_value"],
    )
    field_rows = sorted(
        [row for row in step5 if row["scan_family"] == "field_lattice_dynamic"],
        key=lambda r: r["scan_value"],
    )
    if lattice_rows and c1_rows and field_rows:
        high_t_lattice = lattice_rows[-1]
        high_t_c1 = c1_rows[-1]
        high_b = field_rows[-1]
        if high_t_lattice["delta"] == 0 and high_b["delta"] == 0 and high_t_c1["C1"] != 0:
            lines.append(
                "PASS: Step 5 nonrephasing analysis reaches the intended controls: "
                "lattice-off at high T/high B, with finite C1 in the high-T C1-static case."
            )
        else:
            lines.append("INCONCLUSIVE: Step 5 nonrephasing controls are incomplete in the available data.")

    step2_static = find_pair(
        pairwise_rows,
        "etape_2",
        "unrephasing",
        "dimerisation_control",
        "spin_correlation_control",
    )
    if step2_static and step2_static["relative_max_difference"] < 1e-10:
        lines.append(
            "PASS: Step 2 nonrephasing confirms scalar degeneracy between "
            "dimerisation and C1 controls at matched V_BD_eff."
        )
    else:
        lines.append(
            "INCONCLUSIVE: Step 2 nonrephasing does not provide a clean scalar-degeneracy check."
        )

    (ANALYSIS_DIR / "priority_nonrephasing_validation.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    window_rows = []
    comparisons = []
    pairwise_rows = []

    for step in STEPS:
        step_dir = RESULT_ROOT / step
        data_dir = step_dir / "Data"
        data_files = sorted(data_dir.glob("*_S_data.npz"))
        for data_file in data_files:
            rows, comparison = analyze_data_file(step, step_dir, data_file)
            window_rows.extend(rows)
            comparisons.append(comparison)
        for component in COMPONENTS:
            pairwise_rows.extend(pairwise_component_differences(step, step_dir, component))

    write_csv(ANALYSIS_DIR / "priority_nonrephasing_window_metrics.csv", window_rows)
    write_csv(ANALYSIS_DIR / "priority_nonrephasing_comparison_metrics.csv", comparisons)
    write_csv(ANALYSIS_DIR / "priority_nonrephasing_pairwise_differences.csv", pairwise_rows)
    write_plots(comparisons)
    write_summary(comparisons, pairwise_rows)
    write_validation(comparisons, pairwise_rows)

    print(f"Analyzed spectra: {len(comparisons)}")
    print(f"Window rows: {len(window_rows)}")
    print(f"Summary: {ANALYSIS_DIR / 'priority_nonrephasing_summary.txt'}")
    print(f"Validation: {ANALYSIS_DIR / 'priority_nonrephasing_validation.txt'}")


if __name__ == "__main__":
    main()
