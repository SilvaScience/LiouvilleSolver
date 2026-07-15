from pathlib import Path
import csv
import json
import math

import numpy as np


RESULT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = RESULT_ROOT / "Data"
SUMMARIES_DIR = RESULT_ROOT / "Summaries"
ANALYSIS_DIR = RESULT_ROOT / "Analysis"

COMPONENT = "S_component_rephasing"
WINDOW_HALF_WIDTH_EV = 0.025


def eigen_energies(delta_dark, delta_bright, v_static):
    center = 0.5 * (delta_dark + delta_bright)
    split = math.sqrt((0.5 * (delta_bright - delta_dark)) ** 2 + v_static**2)
    return center - split, center + split


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


def read_summary(data_file):
    stem = data_file.name.replace("_S_data.npz", "")
    summary_path = SUMMARIES_DIR / f"{stem}_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def window_definitions(e_minus, e_plus, omega_q):
    windows = {
        "diag_minus": (-e_minus, e_minus),
        "diag_plus": (-e_plus, e_plus),
        "cross_plus_minus": (-e_plus, e_minus),
        "cross_minus_plus": (-e_minus, e_plus),
    }
    for label, (omega1, omega3) in list(windows.items()):
        windows[f"{label}_det_plus_Q"] = (omega1, omega3 + omega_q)
        windows[f"{label}_det_minus_Q"] = (omega1, omega3 - omega_q)
        windows[f"{label}_coh_plus_Q"] = (omega1 - omega_q, omega3)
        windows[f"{label}_coh_minus_Q"] = (omega1 + omega_q, omega3)
    return windows


def analyze_file(data_file):
    summary = read_summary(data_file)
    model = summary["model_parameters"]
    derived = summary["derived"]
    data = np.load(data_file)
    omega1 = data["omega1"]
    omega3 = data["omega3"]
    matrix = data[COMPONENT]

    e_minus, e_plus = eigen_energies(
        float(model["Delta_dark"]),
        float(model["Delta_Bright"]),
        float(derived["V_BD_static"]),
    )
    windows = window_definitions(e_minus, e_plus, float(derived["omega_Q"]))

    rows = []
    for window, (center_omega1, center_omega3) in windows.items():
        metrics = integrate_window(matrix, omega1, omega3, center_omega1, center_omega3)
        rows.append(
            {
                "case": summary["active_case_key"],
                "run_label": summary["run_label"],
                "file": data_file.name,
                "T": float(derived["T"]),
                "B": float(derived["B"]),
                "T_SP": float(derived["T_SP"]),
                "delta": float(derived["delta"]),
                "C1": float(derived["C1"]),
                "V_BD_static": float(derived["V_BD_static"]),
                "omega_Q": float(derived["omega_Q"]),
                "g_Q_eff": float(derived["g_Q_eff"]),
                "lambda_delta": float(model["lambda_delta"]),
                "lambda_C": float(model["lambda_C"]),
                "E_minus": e_minus,
                "E_plus": e_plus,
                "window": window,
                "center_omega1": center_omega1,
                "center_omega3": center_omega3,
                "window_half_width_eV": WINDOW_HALF_WIDTH_EV,
                **metrics,
            }
        )
    return rows


def add_group_metrics(rows):
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["window"]] = row
    enriched = []
    for row in rows:
        out = dict(row)
        case_rows = by_case[row["case"]]
        diag_main = case_rows["diag_minus"]["abs_area"] + case_rows["diag_plus"]["abs_area"]
        cross_main = (
            case_rows["cross_plus_minus"]["abs_area"]
            + case_rows["cross_minus_plus"]["abs_area"]
        )
        side_cross = 0.0
        for name in (
            "cross_plus_minus_det_plus_Q",
            "cross_plus_minus_det_minus_Q",
            "cross_plus_minus_coh_plus_Q",
            "cross_plus_minus_coh_minus_Q",
            "cross_minus_plus_det_plus_Q",
            "cross_minus_plus_det_minus_Q",
            "cross_minus_plus_coh_plus_Q",
            "cross_minus_plus_coh_minus_Q",
        ):
            side_cross += case_rows[name]["abs_area"]
        out["diag_abs_area_sum"] = diag_main
        out["cross_abs_area_sum"] = cross_main
        out["sideband_cross_abs_area_sum"] = side_cross
        out["cross_to_diag_ratio"] = cross_main / diag_main if diag_main else np.nan
        out["sideband_to_main_cross_ratio"] = side_cross / cross_main if cross_main else np.nan
        enriched.append(out)
    return enriched


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pairwise_differences():
    data_files = {
        read_summary(path)["active_case_key"]: path
        for path in sorted(DATA_DIR.glob("*_S_data.npz"))
    }
    rows = []
    cases = sorted(data_files)
    for i, case_a in enumerate(cases):
        for case_b in cases[i + 1 :]:
            data_a = np.load(data_files[case_a])
            data_b = np.load(data_files[case_b])
            matrix_a = data_a[COMPONENT]
            matrix_b = data_b[COMPONENT]
            diff = matrix_a - matrix_b
            denom = max(float(np.max(np.abs(matrix_a))), float(np.max(np.abs(matrix_b))), 1e-300)
            rows.append(
                {
                    "case_a": case_a,
                    "case_b": case_b,
                    "max_abs_difference": float(np.max(np.abs(diff))),
                    "relative_max_difference": float(np.max(np.abs(diff)) / denom),
                    "l2_difference": float(np.linalg.norm(diff.ravel())),
                    "same_shape": matrix_a.shape == matrix_b.shape,
                }
            )
    return rows


def ordered_main_rows(rows):
    order = [
        "lowT_lattice_dynamic",
        "highT_lattice_off_C1_finite",
        "highB_lattice_suppressed",
        "highT_C1_static",
    ]
    by_case = {
        row["case"]: row
        for row in rows
        if row["window"] == "cross_plus_minus"
    }
    return [by_case[case] for case in order if case in by_case]


def write_plots(rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (ANALYSIS_DIR / "plots_skipped.txt").write_text(
            f"Matplotlib plots skipped: {exc!r}\n",
            encoding="utf-8",
        )
        return

    main_rows = ordered_main_rows(rows)
    labels = [row["case"].replace("_", "\n") for row in main_rows]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
    axes = axes.ravel()
    plot_specs = [
        ("delta", "delta(T,B)"),
        ("C1", "C1(T,B)"),
        ("g_Q_eff", "g_Q_eff"),
        ("V_BD_static", "V_BD_static"),
        ("cross_abs_area_sum", "Main cross |S| area"),
        ("sideband_to_main_cross_ratio", "Sideband/main cross"),
    ]
    for ax, (key, title) in zip(axes, plot_specs):
        ax.bar(labels, [row[key] for row in main_rows])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=8)
    fig.savefig(ANALYSIS_DIR / "step4_trend_metrics.png", dpi=200)
    plt.close(fig)

    data_files = sorted(DATA_DIR.glob("*_S_data.npz"), key=lambda p: read_summary(p)["active_case_key"])
    fig, axes = plt.subplots(
        1,
        len(data_files),
        figsize=(4.7 * len(data_files) + 0.8, 4.0),
        constrained_layout=True,
    )
    if len(data_files) == 1:
        axes = [axes]
    vmax = 0.0
    loaded = []
    for path in data_files:
        summary = read_summary(path)
        data = np.load(path)
        matrix = np.real(data[COMPONENT])
        loaded.append((summary["active_case_key"], data["omega1"], data["omega3"], matrix))
        vmax = max(vmax, float(np.max(np.abs(matrix))))
    for ax, (case, omega1, omega3, matrix) in zip(axes, loaded):
        image = ax.imshow(
            matrix.T,
            origin="lower",
            aspect="auto",
            extent=[omega1[0], omega1[-1], omega3[0], omega3[-1]],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(case.replace("_", "\n"), fontsize=9)
        ax.set_xlabel("omega1 (eV)")
        ax.set_ylabel("omega3 (eV)")
    fig.colorbar(image, ax=list(axes), shrink=0.82, pad=0.02, label="Re S")
    fig.savefig(ANALYSIS_DIR / "step4_rephasing_real_comparison.png", dpi=200)
    plt.close(fig)


def find_diff(diff_rows, case_a, case_b):
    target = {case_a, case_b}
    return next(row for row in diff_rows if {row["case_a"], row["case_b"]} == target)


def write_summary(rows, diff_rows):
    main_rows = {row["case"]: row for row in ordered_main_rows(rows)}
    lowT = main_rows.get("lowT_lattice_dynamic")
    highT_off = main_rows.get("highT_lattice_off_C1_finite")
    highB_off = main_rows.get("highB_lattice_suppressed")
    highT_C1 = main_rows.get("highT_C1_static")

    lines = []
    lines.append("Step 4 analysis: independent temperature and field trends")
    lines.append("==========================================================")
    lines.append(f"Result root: {RESULT_ROOT}")
    lines.append(f"Component: {COMPONENT}")
    lines.append(f"Window half-width (eV): {WINDOW_HALF_WIDTH_EV}")
    lines.append("")
    lines.append("Main metrics")
    lines.append("------------")
    for case, row in main_rows.items():
        lines.append(
            f"{case}: T={row['T']:.6g}, B={row['B']:.6g}, "
            f"T_SP={row['T_SP']:.6g}, delta={row['delta']:.8g}, "
            f"C1={row['C1']:.8g}, g_Q_eff={row['g_Q_eff']:.8g}, "
            f"V_BD_static={row['V_BD_static']:.8g}, "
            f"cross_abs_area_sum={row['cross_abs_area_sum']:.8g}, "
            f"cross_to_diag_ratio={row['cross_to_diag_ratio']:.8g}, "
            f"sideband_to_main_cross_ratio={row['sideband_to_main_cross_ratio']:.8g}"
        )
    lines.append("")
    lines.append("Pairwise spectrum differences")
    lines.append("-----------------------------")
    for row in diff_rows:
        lines.append(
            f"{row['case_a']} vs {row['case_b']}: "
            f"max_abs_difference={row['max_abs_difference']:.8g}, "
            f"relative_max_difference={row['relative_max_difference']:.8g}, "
            f"l2_difference={row['l2_difference']:.8g}"
        )
    if lowT and highT_off and highB_off and highT_C1:
        lines.append("")
        lines.append("Interpretation")
        lines.append("--------------")
        lines.append(
            "The low-temperature lattice case has finite delta and finite g_Q_eff. "
            "The high-temperature and high-field lattice-off controls have delta = 0 "
            "and g_Q_eff = 0 while C1 remains finite."
        )
        lines.append(
            f"The main cross-window area changes from {lowT['cross_abs_area_sum']:.8g} "
            f"to {highT_off['cross_abs_area_sum']:.8g} in the high-temperature "
            "lattice-off control."
        )
        lines.append(
            f"In the high-field lattice-suppressed control the same metric is "
            f"{highB_off['cross_abs_area_sum']:.8g}."
        )
        lines.append(
            f"The high-temperature C1-static control keeps delta = 0 but gives "
            f"cross_abs_area_sum={highT_C1['cross_abs_area_sum']:.8g}; any feature "
            "that survives in this control is not unique to the dimerised phase."
        )
    (ANALYSIS_DIR / "step4_analysis_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_validation_outcome(rows, diff_rows):
    main_rows = {row["case"]: row for row in ordered_main_rows(rows)}
    required = {
        "lowT_lattice_dynamic",
        "highT_lattice_off_C1_finite",
        "highB_lattice_suppressed",
        "highT_C1_static",
    }
    lines = []
    if not required.issubset(main_rows):
        missing = sorted(required - set(main_rows))
        lines.append(f"INCOMPLETE: missing cases {missing}.")
    else:
        lowT = main_rows["lowT_lattice_dynamic"]
        highT_off = main_rows["highT_lattice_off_C1_finite"]
        highB_off = main_rows["highB_lattice_suppressed"]
        highT_C1 = main_rows["highT_C1_static"]

        if lowT["delta"] > 0 and lowT["g_Q_eff"] > 0:
            lines.append("PASS: The low-temperature lattice reference has finite delta and g_Q_eff.")
        else:
            lines.append("FAIL: The low-temperature lattice reference does not activate the lattice channel.")

        if highT_off["delta"] == 0 and highT_off["g_Q_eff"] == 0 and highT_off["C1"] != 0:
            lines.append("PASS: The high-temperature lattice-off control has delta = 0, g_Q_eff = 0, and finite C1.")
        else:
            lines.append("FAIL: The high-temperature lattice-off control does not isolate finite C1 from delta.")

        if highB_off["delta"] == 0 and highB_off["g_Q_eff"] == 0 and highB_off["C1"] != 0:
            lines.append("PASS: The high-field control suppresses the lattice channel while keeping finite C1.")
        else:
            lines.append("FAIL: The high-field control does not suppress the lattice channel cleanly.")

        if highT_C1["delta"] == 0 and highT_C1["C1"] != 0 and highT_C1["V_BD_static"] > highT_off["V_BD_static"]:
            lines.append(
                "PASS: The high-temperature C1-static control produces a persistent scalar mixing channel "
                "with delta = 0."
            )
        else:
            lines.append("INCONCLUSIVE: The C1-static control is not clearly separated from the lattice-off baseline.")

        diff_low_highT = find_diff(diff_rows, "lowT_lattice_dynamic", "highT_lattice_off_C1_finite")
        diff_highT_C1 = find_diff(diff_rows, "highT_lattice_off_C1_finite", "highT_C1_static")
        lines.append(
            f"LowT lattice vs highT lattice-off relative max difference: "
            f"{diff_low_highT['relative_max_difference']:.3e}."
        )
        lines.append(
            f"HighT lattice-off vs highT C1-static relative max difference: "
            f"{diff_highT_C1['relative_max_difference']:.3e}."
        )
    (ANALYSIS_DIR / "validation_outcome.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    data_files = sorted(DATA_DIR.glob("*_S_data.npz"))
    rows = []
    for data_file in data_files:
        rows.extend(analyze_file(data_file))
    rows = add_group_metrics(rows)
    diff_rows = pairwise_differences()

    write_csv(ANALYSIS_DIR / "step4_window_metrics.csv", rows)
    write_csv(ANALYSIS_DIR / "step4_pairwise_differences.csv", diff_rows)
    write_plots(rows)
    write_summary(rows, diff_rows)
    write_validation_outcome(rows, diff_rows)

    print(f"Analyzed files: {len(data_files)}")
    print(f"CSV: {ANALYSIS_DIR / 'step4_window_metrics.csv'}")
    print(f"Summary: {ANALYSIS_DIR / 'step4_analysis_summary.txt'}")
    print(f"Validation: {ANALYSIS_DIR / 'validation_outcome.txt'}")


if __name__ == "__main__":
    main()
