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
    derived = summary["derived"]
    data = np.load(data_file)
    omega1 = data["omega1"]
    omega3 = data["omega3"]
    matrix = data[COMPONENT]
    windows = window_definitions(
        float(derived["E_minus"]),
        float(derived["E_plus"]),
        float(derived["omega_Q"]),
    )

    rows = []
    for window, (center_omega1, center_omega3) in windows.items():
        metrics = integrate_window(matrix, omega1, omega3, center_omega1, center_omega3)
        rows.append(
            {
                "case": summary["active_case_key"],
                "run_label": summary["run_label"],
                "file": data_file.name,
                "scan_family": summary["scan_family"],
                "scan_variable": summary["scan_variable"],
                "scan_value": float(summary["scan_value"]),
                "V_BD_static": float(derived["V_BD_static"]),
                "omega_Q": float(derived["omega_Q"]),
                "g_Q": float(derived["g_Q_eff"]),
                "E_minus": float(derived["E_minus"]),
                "E_plus": float(derived["E_plus"]),
                "rephasing_max_abs_full": float(summary["numerical_observables"]["rephasing_max_abs"]),
                "rephasing_l2_norm_full": float(summary["numerical_observables"]["rephasing_l2_norm"]),
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


def scan_metrics(rows):
    main_rows = [row for row in rows if row["window"] == "cross_plus_minus"]
    return sorted(main_rows, key=lambda row: (row["scan_family"], row["scan_value"]))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def family_rows(rows, family):
    return sorted(
        [row for row in rows if row["scan_family"] == family],
        key=lambda row: row["scan_value"],
    )


def write_plots(metrics):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (ANALYSIS_DIR / "plots_skipped.txt").write_text(
            f"Matplotlib plots skipped: {exc!r}\n",
            encoding="utf-8",
        )
        return

    for family, xkey, xlabel in (
        ("gQ_scan", "g_Q", "g_Q"),
        ("omegaQ_scan", "omega_Q", "omega_Q (eV)"),
    ):
        rows = family_rows(metrics, family)
        if not rows:
            continue
        x = np.array([row[xkey] for row in rows], dtype=float)
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
        axes = axes.ravel()
        specs = [
            ("cross_abs_area_sum", "Main cross |S| area"),
            ("cross_to_diag_ratio", "Cross/diagonal"),
            ("sideband_to_main_cross_ratio", "Sideband/main cross"),
            ("rephasing_l2_norm_full", "Full rephasing L2 norm"),
        ]
        for ax, (key, title) in zip(axes, specs):
            ax.plot(x, [row[key] for row in rows], marker="o")
            ax.set_xlabel(xlabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
        fig.savefig(ANALYSIS_DIR / f"step6_{family}_metrics.png", dpi=200)
        plt.close(fig)


def write_summary(metrics):
    lines = []
    lines.append("Step 6 analysis: dynamic phonon parameter scans")
    lines.append("================================================")
    lines.append(f"Result root: {RESULT_ROOT}")
    lines.append(f"Component: {COMPONENT}")
    lines.append(f"Window half-width (eV): {WINDOW_HALF_WIDTH_EV}")
    lines.append("")

    for family in ("gQ_scan", "omegaQ_scan"):
        rows = family_rows(metrics, family)
        lines.append(family)
        lines.append("-" * len(family))
        for row in rows:
            lines.append(
                f"{row['scan_variable']}={row['scan_value']:.6g}: "
                f"V_BD_static={row['V_BD_static']:.8g}, "
                f"g_Q={row['g_Q']:.8g}, omega_Q={row['omega_Q']:.8g}, "
                f"cross_abs_area_sum={row['cross_abs_area_sum']:.8g}, "
                f"cross_to_diag_ratio={row['cross_to_diag_ratio']:.8g}, "
                f"sideband_to_main_cross_ratio={row['sideband_to_main_cross_ratio']:.8g}, "
                f"rephasing_l2_norm_full={row['rephasing_l2_norm_full']:.8g}"
            )
        lines.append("")
    lines.append("Interpretation")
    lines.append("--------------")
    lines.append(
        "Compare the g_Q=0 static reference against nonzero g_Q values at fixed "
        "V_BD_static and omega_Q.  Then compare omega_Q points at fixed g_Q."
    )
    (ANALYSIS_DIR / "step6_analysis_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_validation_outcome(metrics):
    lines = []
    gq_rows = family_rows(metrics, "gQ_scan")
    omega_rows = family_rows(metrics, "omegaQ_scan")

    if len(gq_rows) >= 2:
        reference = min(gq_rows, key=lambda row: abs(row["g_Q"]))
        max_diff = max(
            abs(row["rephasing_l2_norm_full"] - reference["rephasing_l2_norm_full"])
            for row in gq_rows
        )
        rel = max_diff / max(abs(reference["rephasing_l2_norm_full"]), 1e-300)
        if rel > 1e-4:
            lines.append(
                f"PASS: nonzero g_Q changes the full rephasing norm relative to g_Q=0 "
                f"(relative change {rel:.3e})."
            )
        else:
            lines.append(
                f"INCONCLUSIVE: g_Q scan changes are small relative to g_Q=0 "
                f"(relative change {rel:.3e})."
            )
    else:
        lines.append("INCOMPLETE: g_Q scan has fewer than two points.")

    if len(omega_rows) >= 2:
        values = [row["sideband_to_main_cross_ratio"] for row in omega_rows]
        spread = max(values) - min(values)
        scale = max(max(abs(v) for v in values), 1e-300)
        rel = spread / scale
        if rel > 1e-3:
            lines.append(
                f"PASS: sideband/main-cross ratio varies across omega_Q "
                f"(relative spread {rel:.3e})."
            )
        else:
            lines.append(
                f"INCONCLUSIVE: sideband/main-cross ratio is nearly unchanged across omega_Q "
                f"(relative spread {rel:.3e})."
            )
    else:
        lines.append("INCOMPLETE: omega_Q scan has fewer than two points.")

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
    metrics = scan_metrics(rows)

    write_csv(ANALYSIS_DIR / "step6_window_metrics.csv", rows)
    write_csv(ANALYSIS_DIR / "step6_scan_metrics.csv", metrics)
    write_plots(metrics)
    write_summary(metrics)
    write_validation_outcome(metrics)

    print(f"Analyzed files: {len(data_files)}")
    print(f"Scan metrics: {ANALYSIS_DIR / 'step6_scan_metrics.csv'}")
    print(f"Summary: {ANALYSIS_DIR / 'step6_analysis_summary.txt'}")
    print(f"Validation: {ANALYSIS_DIR / 'validation_outcome.txt'}")


if __name__ == "__main__":
    main()
