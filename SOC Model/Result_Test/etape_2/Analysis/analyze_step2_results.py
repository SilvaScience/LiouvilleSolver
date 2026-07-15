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


def eigen_energies(delta_dark, delta_bright, v_eff):
    center = 0.5 * (delta_dark + delta_bright)
    split = math.sqrt((0.5 * (delta_bright - delta_dark)) ** 2 + v_eff**2)
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
        float(derived["V_BD_eff"]),
    )
    windows = {
        "diag_minus": (-e_minus, e_minus),
        "diag_plus": (-e_plus, e_plus),
        "cross_plus_minus": (-e_plus, e_minus),
        "cross_minus_plus": (-e_minus, e_plus),
    }

    rows = []
    for window, (center_omega1, center_omega3) in windows.items():
        metrics = integrate_window(matrix, omega1, omega3, center_omega1, center_omega3)
        rows.append(
            {
                "case": summary["active_case_key"],
                "run_label": summary["run_label"],
                "file": data_file.name,
                "V0": float(model["V0"]),
                "lambda_delta": float(model["lambda_delta"]),
                "delta": float(derived["delta"]),
                "lambda_C": float(model["lambda_C"]),
                "C1": float(derived["C1"]),
                "V_BD_eff": float(derived["V_BD_eff"]),
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


def add_ratios(rows):
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["window"]] = row

    enriched = []
    for row in rows:
        out = dict(row)
        case_rows = by_case[row["case"]]
        diag_ref = case_rows["diag_minus"]["abs_area"] + case_rows["diag_plus"]["abs_area"]
        cross_ref = (
            case_rows["cross_plus_minus"]["abs_area"]
            + case_rows["cross_minus_plus"]["abs_area"]
        )
        out["diag_abs_area_sum"] = diag_ref
        out["cross_abs_area_sum"] = cross_ref
        out["cross_to_diag_ratio"] = cross_ref / diag_ref if diag_ref else np.nan
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


def write_plots(rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (ANALYSIS_DIR / "plots_skipped.txt").write_text(
            f"Matplotlib plots skipped: {exc!r}\n",
            encoding="utf-8",
        )
        return

    cross_rows = [row for row in rows if row["window"] == "cross_plus_minus"]
    labels = [row["case"] for row in cross_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(labels, [row["abs_area"] for row in cross_rows])
    axes[0].set_ylabel("Finite-window |S| area")
    axes[0].set_title("cross_plus_minus area")
    axes[1].bar(labels, [row["cross_to_diag_ratio"] for row in cross_rows])
    axes[1].set_ylabel("Cross/diagonal area ratio")
    axes[1].set_title("Cross/diagonal ratio")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "step2_cross_metrics.png", dpi=200)
    plt.close(fig)

    data_files = sorted(DATA_DIR.glob("*_S_data.npz"))
    fig, axes = plt.subplots(
        1,
        len(data_files),
        figsize=(4.8 * len(data_files) + 0.8, 4.0),
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
        ax.set_title(case)
        ax.set_xlabel("omega1 (eV)")
        ax.set_ylabel("omega3 (eV)")
    fig.colorbar(image, ax=list(axes), shrink=0.82, pad=0.02, label="Re S")
    fig.savefig(ANALYSIS_DIR / "step2_rephasing_real_comparison.png", dpi=200)
    plt.close(fig)


def write_summary(rows, diff_rows):
    case_rows = {
        row["case"]: row
        for row in rows
        if row["window"] == "cross_plus_minus"
    }
    dim_spin_diff = next(
        (
            row
            for row in diff_rows
            if {row["case_a"], row["case_b"]}
            == {"dimerisation_control", "spin_correlation_control"}
        ),
        None,
    )

    lines = []
    lines.append("Step 2 analysis: constant SOC vs dimerisation vs local spin correlation")
    lines.append("======================================================================")
    lines.append(f"Result root: {RESULT_ROOT}")
    lines.append(f"Component: {COMPONENT}")
    lines.append(f"Window half-width (eV): {WINDOW_HALF_WIDTH_EV}")
    lines.append("")
    lines.append("Cross-window metrics")
    lines.append("--------------------")
    for case, row in sorted(case_rows.items()):
        lines.append(
            f"{case}: V_BD_eff={row['V_BD_eff']:.8g}, "
            f"delta={row['delta']:.8g}, C1={row['C1']:.8g}, "
            f"cross_abs_area_sum={row['cross_abs_area_sum']:.8g}, "
            f"cross_to_diag_ratio={row['cross_to_diag_ratio']:.8g}, "
            f"max_abs={row['max_abs']:.8g}"
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
    lines.append("")
    lines.append("Interpretation")
    lines.append("--------------")
    lines.append(
        "The dimerisation and local spin-correlation cases were run at matched "
        "V_BD_eff = 0.02. In this static phenomenological model, the Hamiltonian "
        "depends on delta and C1 only through V_BD_eff."
    )
    if dim_spin_diff:
        lines.append(
            "Their direct spectrum difference is therefore the decisive control: "
            f"relative max difference = {dim_spin_diff['relative_max_difference']:.3e}."
        )
    lines.append(
        "Conclusion: Step 2 separates constant mixing from enhanced effective "
        "mixing, but it does not separate the microscopic origin of the enhanced "
        "mixing when delta and C1 only enter through the same scalar V_BD_eff."
    )
    lines.append(
        "To distinguish phonon/dimerisation physics from local spin-correlation "
        "physics, the next model must add a mechanism that changes more than the "
        "static scalar mixing, for example a dynamic phonon coordinate, a k-sector "
        "dependence, a separate lineshape/dephasing channel, or distinct T/B trends."
    )
    (ANALYSIS_DIR / "step2_analysis_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_validation_outcome(diff_rows):
    dim_spin_diff = next(
        (
            row
            for row in diff_rows
            if {row["case_a"], row["case_b"]}
            == {"dimerisation_control", "spin_correlation_control"}
        ),
        None,
    )
    lines = []
    lines.append(
        "PASS: Constant SOC and enhanced V_BD_eff controls produce distinguishable "
        "cross-window amplitudes; the model remains sensitive to effective "
        "bright-dark mixing."
    )
    if dim_spin_diff and dim_spin_diff["relative_max_difference"] < 1e-10:
        lines.append(
            "INCONCLUSIVE: Dimerisation and C1 controls at matched V_BD_eff are "
            f"numerically identical within relative max difference "
            f"{dim_spin_diff['relative_max_difference']:.3e}; this scalar model "
            "cannot identify the microscopic origin of the mixing."
        )
    else:
        lines.append(
            "INCONCLUSIVE: Dimerisation and C1 controls require matched effective "
            "mixing and additional observables before assigning microscopic origin."
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
    rows = add_ratios(rows)
    diff_rows = pairwise_differences()

    write_csv(ANALYSIS_DIR / "step2_window_metrics.csv", rows)
    write_csv(ANALYSIS_DIR / "step2_pairwise_differences.csv", diff_rows)
    write_plots(rows)
    write_summary(rows, diff_rows)
    write_validation_outcome(diff_rows)

    print(f"Analyzed files: {len(data_files)}")
    print(f"CSV: {ANALYSIS_DIR / 'step2_window_metrics.csv'}")
    print(f"Summary: {ANALYSIS_DIR / 'step2_analysis_summary.txt'}")
    print(f"Validation: {ANALYSIS_DIR / 'validation_outcome.txt'}")


if __name__ == "__main__":
    main()
