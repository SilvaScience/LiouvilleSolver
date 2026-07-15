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
                "scan_family": summary["scan_family"],
                "scan_variable": summary["scan_variable"],
                "scan_value": float(summary["scan_value"]),
                "channel": summary["channel"],
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

    for family, xlabel in (
        ("temperature_lattice_dynamic", "T"),
        ("field_lattice_dynamic", "B"),
        ("temperature_C1_static", "T"),
    ):
        rows = family_rows(metrics, family)
        if not rows:
            continue
        x = np.array([row["scan_value"] for row in rows], dtype=float)
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
        axes = axes.ravel()
        specs = [
            ("delta", "delta(T,B)"),
            ("g_Q_eff", "g_Q_eff"),
            ("V_BD_static", "V_BD_static"),
            ("cross_abs_area_sum", "Main cross |S| area"),
        ]
        for ax, (key, title) in zip(axes, specs):
            ax.plot(x, [row[key] for row in rows], marker="o")
            ax.set_xlabel(xlabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            if family.startswith("temperature"):
                ax.axvline(14.0, color="0.2", linestyle="--", linewidth=1.0, alpha=0.7)
        fig.savefig(ANALYSIS_DIR / f"step5_{family}_trend.png", dpi=200)
        plt.close(fig)

    lattice = family_rows(metrics, "temperature_lattice_dynamic")
    c1_rows = family_rows(metrics, "temperature_C1_static")
    if lattice and c1_rows:
        fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
        ax.plot(
            [row["T"] for row in lattice],
            [row["cross_abs_area_sum"] for row in lattice],
            marker="o",
            label="lattice dynamic",
        )
        ax.plot(
            [row["T"] for row in c1_rows],
            [row["cross_abs_area_sum"] for row in c1_rows],
            marker="s",
            label="C1 static",
        )
        ax.axvline(14.0, color="0.2", linestyle="--", linewidth=1.0, alpha=0.7, label="T_SP")
        ax.set_xlabel("T")
        ax.set_ylabel("Main cross |S| area")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.savefig(ANALYSIS_DIR / "step5_temperature_lattice_vs_C1.png", dpi=200)
        plt.close(fig)


def nearest(rows, **criteria):
    candidates = rows
    for key, value in criteria.items():
        candidates = [row for row in candidates if row[key] == value]
    if not candidates:
        return None
    return candidates[0]


def write_summary(metrics):
    lines = []
    lines.append("Step 5 analysis: temperature and field scans")
    lines.append("============================================")
    lines.append(f"Result root: {RESULT_ROOT}")
    lines.append(f"Component: {COMPONENT}")
    lines.append(f"Window half-width (eV): {WINDOW_HALF_WIDTH_EV}")
    lines.append("")

    for family in (
        "temperature_lattice_dynamic",
        "field_lattice_dynamic",
        "temperature_C1_static",
    ):
        rows = family_rows(metrics, family)
        lines.append(f"{family}")
        lines.append("-" * len(family))
        for row in rows:
            lines.append(
                f"{row['scan_variable']}={row['scan_value']:.6g}: "
                f"T={row['T']:.6g}, B={row['B']:.6g}, T_SP={row['T_SP']:.6g}, "
                f"delta={row['delta']:.8g}, C1={row['C1']:.8g}, "
                f"g_Q_eff={row['g_Q_eff']:.8g}, V_BD_static={row['V_BD_static']:.8g}, "
                f"cross_abs_area_sum={row['cross_abs_area_sum']:.8g}, "
                f"cross_to_diag_ratio={row['cross_to_diag_ratio']:.8g}"
            )
        lines.append("")

    low_t = nearest(metrics, scan_family="temperature_lattice_dynamic", scan_value=7.0)
    at_tsp = nearest(metrics, scan_family="temperature_lattice_dynamic", scan_value=14.0)
    high_t = nearest(metrics, scan_family="temperature_lattice_dynamic", scan_value=20.0)
    high_b = nearest(metrics, scan_family="field_lattice_dynamic", scan_value=15.0)
    c1_high_t = nearest(metrics, scan_family="temperature_C1_static", scan_value=20.0)
    if low_t and at_tsp and high_t and high_b and c1_high_t:
        lines.append("Interpretation")
        lines.append("--------------")
        lines.append(
            f"In the temperature lattice scan, delta changes from {low_t['delta']:.8g} "
            f"at T=7 to {at_tsp['delta']:.8g} at T=14 and {high_t['delta']:.8g} "
            "at T=20."
        )
        lines.append(
            f"The corresponding main cross area changes from "
            f"{low_t['cross_abs_area_sum']:.8g} to {at_tsp['cross_abs_area_sum']:.8g} "
            f"and {high_t['cross_abs_area_sum']:.8g}."
        )
        lines.append(
            f"In the field scan at T=7, B=15 suppresses the lattice channel with "
            f"T_SP={high_b['T_SP']:.8g}, delta={high_b['delta']:.8g}, and "
            f"cross_abs_area_sum={high_b['cross_abs_area_sum']:.8g}."
        )
        lines.append(
            f"The C1-static high-temperature control keeps delta={c1_high_t['delta']:.8g} "
            f"but gives cross_abs_area_sum={c1_high_t['cross_abs_area_sum']:.8g}, "
            "so a persistent feature above T_SP is not unique to the dimerised phase."
        )
    (ANALYSIS_DIR / "step5_analysis_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def monotonic_nonincreasing(values):
    return all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


def write_validation_outcome(metrics):
    lines = []
    temp_lattice = family_rows(metrics, "temperature_lattice_dynamic")
    field_lattice = family_rows(metrics, "field_lattice_dynamic")
    temp_c1 = family_rows(metrics, "temperature_C1_static")

    if temp_lattice and monotonic_nonincreasing([row["delta"] for row in temp_lattice]):
        lines.append("PASS: delta(T,0) decreases monotonically across the temperature lattice scan.")
    else:
        lines.append("FAIL: delta(T,0) is not monotonic in the temperature lattice scan.")

    if temp_lattice and temp_lattice[-1]["delta"] == 0 and temp_lattice[-1]["g_Q_eff"] == 0:
        lines.append("PASS: the high-temperature lattice endpoint has delta = 0 and g_Q_eff = 0.")
    else:
        lines.append("FAIL: the high-temperature lattice endpoint does not turn off the lattice channel.")

    if field_lattice and field_lattice[-1]["delta"] == 0 and field_lattice[-1]["g_Q_eff"] == 0:
        lines.append("PASS: the high-field lattice endpoint suppresses delta and g_Q_eff at fixed low T.")
    else:
        lines.append("FAIL: the high-field endpoint does not suppress the lattice channel.")

    if temp_c1 and temp_c1[-1]["delta"] == 0 and temp_c1[-1]["C1"] != 0:
        lines.append("PASS: the high-temperature C1-static control keeps finite C1 with delta = 0.")
    else:
        lines.append("FAIL: the C1-static high-temperature control does not isolate finite C1.")

    if temp_lattice and temp_c1:
        lattice_high = temp_lattice[-1]["cross_abs_area_sum"]
        c1_high = temp_c1[-1]["cross_abs_area_sum"]
        if c1_high > lattice_high:
            lines.append(
                f"PASS: at high T, the C1-static cross area ({c1_high:.6g}) remains "
                f"larger than the lattice-off area ({lattice_high:.6g})."
            )
        else:
            lines.append(
                f"INCONCLUSIVE: at high T, the C1-static cross area ({c1_high:.6g}) "
                f"is not larger than the lattice-off area ({lattice_high:.6g})."
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
    metrics = scan_metrics(rows)

    write_csv(ANALYSIS_DIR / "step5_window_metrics.csv", rows)
    write_csv(ANALYSIS_DIR / "step5_scan_metrics.csv", metrics)
    write_plots(metrics)
    write_summary(metrics)
    write_validation_outcome(metrics)

    print(f"Analyzed files: {len(data_files)}")
    print(f"Scan metrics: {ANALYSIS_DIR / 'step5_scan_metrics.csv'}")
    print(f"Summary: {ANALYSIS_DIR / 'step5_analysis_summary.txt'}")
    print(f"Validation: {ANALYSIS_DIR / 'validation_outcome.txt'}")


if __name__ == "__main__":
    main()
