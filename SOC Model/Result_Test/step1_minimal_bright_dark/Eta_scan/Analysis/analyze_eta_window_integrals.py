from pathlib import Path
import csv
import math
import re
import textwrap

import numpy as np


RESULT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent

# Finite integration window around each target peak, in eV.
# Increase this for broad spectra; decrease it to isolate leakage tails.
WINDOW_HALF_WIDTH_EV = 0.025

# Component used for the validation metrics.
DEFAULT_COMPONENT = "S_component_rephasing"


def parse_run_parameters(text_path):
    values = {}
    for line in text_path.read_text(encoding="utf-8").splitlines():
        if ": " not in line:
            continue
        key, raw = line.split(": ", 1)
        if key in {
            "Eta",
            "V0",
            "N_w",
            "Delta_dark",
            "Delta_Bright",
            "mu_D",
            "gamma_orb",
            "gamma_spin",
        }:
            try:
                values[key] = int(raw) if key == "N_w" else float(raw)
            except ValueError:
                values[key] = raw
    return values


def matching_text_file(data_file):
    return data_file.with_name(data_file.name.replace("_S_data.npz", ".txt"))


def run_label_from_name(path):
    match = re.search(
        r"(?:N_w_(?P<N_w>\d+)_)?Eta_(?P<eta>[^_]+)_V0_(?P<v0>[^_]+(?:\.\d+)?)_S_data",
        path.name,
    )
    if match:
        return match.group("eta"), match.group("v0"), match.group("N_w")
    return "", "", None


def eigen_energies(delta_dark, delta_bright, v0):
    center = 0.5 * (delta_dark + delta_bright)
    split = np.sqrt((0.5 * (delta_bright - delta_dark)) ** 2 + v0**2)
    return center - split, center + split


def finite_window_mask(omega1, omega3, center_omega1, center_omega3, half_width):
    mask1 = np.abs(omega1 - center_omega1) <= half_width
    mask3 = np.abs(omega3 - center_omega3) <= half_width
    return mask1, mask3


def integrate_window(matrix, omega1, omega3, center_omega1, center_omega3, half_width):
    mask1, mask3 = finite_window_mask(
        omega1, omega3, center_omega1, center_omega3, half_width
    )
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
        # Fallback for very narrow windows on coarse grids.
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


def analyze_file(data_file):
    text_file = matching_text_file(data_file)
    params = parse_run_parameters(text_file) if text_file.exists() else {}
    eta_from_name, v0_from_name, nw_from_name = run_label_from_name(data_file)

    eta = float(params.get("Eta", eta_from_name))
    v0 = float(params.get("V0", v0_from_name))
    delta_dark = float(params.get("Delta_dark", 0.9))
    delta_bright = float(params.get("Delta_Bright", 1.0))

    data = np.load(data_file)
    omega1 = data["omega1"]
    omega3 = data["omega3"]
    n_w = int(params.get("N_w", nw_from_name or len(omega1)))
    if DEFAULT_COMPONENT not in data.files:
        raise KeyError(f"{data_file} does not contain {DEFAULT_COMPONENT}")
    matrix = data[DEFAULT_COMPONENT]

    e_minus, e_plus = eigen_energies(delta_dark, delta_bright, v0)
    windows = {
        "diag_minus": (-e_minus, e_minus),
        "diag_plus": (-e_plus, e_plus),
        "cross_plus_minus": (-e_plus, e_minus),
        "cross_minus_plus": (-e_minus, e_plus),
    }

    rows = []
    for window_name, (center_w1, center_w3) in windows.items():
        metrics = integrate_window(
            matrix, omega1, omega3, center_w1, center_w3, WINDOW_HALF_WIDTH_EV
        )
        rows.append(
            {
                "file": data_file.name,
                "eta": eta,
                "V0": v0,
                "N_w": n_w,
                "Delta_dark": delta_dark,
                "Delta_Bright": delta_bright,
                "E_minus": e_minus,
                "E_plus": e_plus,
                "window": window_name,
                "center_omega1": center_w1,
                "center_omega3": center_w3,
                "window_half_width_eV": WINDOW_HALF_WIDTH_EV,
                **metrics,
            }
        )
    return rows


def add_background_subtraction(rows):
    exact_baseline = {}
    eta_window_baseline = {}
    for row in rows:
        if abs(row["V0"]) < 1e-15:
            exact_baseline[(row["eta"], row["N_w"], row["window"])] = row
            eta_window_baseline.setdefault((row["eta"], row["window"]), []).append(row)

    enriched = []
    for row in rows:
        out = dict(row)
        base = exact_baseline.get((row["eta"], row["N_w"], row["window"]))
        background_mode = "same_eta_same_N_w"
        if base is None:
            candidates = eta_window_baseline.get((row["eta"], row["window"]), [])
            if candidates:
                base = min(candidates, key=lambda item: abs(item["N_w"] - row["N_w"]))
                background_mode = "same_eta_closest_N_w"
        if base is None:
            out["background_available"] = False
            out["background_mode"] = "missing"
            out["baseline_N_w"] = ""
            out["baseline_file"] = ""
            out["abs_area_net"] = np.nan
            out["max_abs_net"] = np.nan
            out["complex_area_real_net"] = np.nan
            out["complex_area_imag_net"] = np.nan
        else:
            out["background_available"] = True
            out["background_mode"] = background_mode
            out["baseline_N_w"] = base["N_w"]
            out["baseline_file"] = base["file"]
            out["abs_area_net"] = row["abs_area"] - base["abs_area"]
            out["max_abs_net"] = row["max_abs"] - base["max_abs"]
            out["complex_area_real_net"] = (
                row["complex_area_real"] - base["complex_area_real"]
            )
            out["complex_area_imag_net"] = (
                row["complex_area_imag"] - base["complex_area_imag"]
            )
        enriched.append(out)
    return enriched


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, rows):
    etas = sorted({row["eta"] for row in rows})
    v0s = sorted({row["V0"] for row in rows})
    nws = sorted({row["N_w"] for row in rows})
    cross_rows = [
        row for row in rows if row["window"] in {"cross_plus_minus", "cross_minus_plus"}
    ]

    lines = []
    lines.append("Step 1 Eta-scan finite-window analysis")
    lines.append("======================================")
    lines.append(f"Result root: {RESULT_ROOT}")
    lines.append(f"Component: {DEFAULT_COMPONENT}")
    lines.append(f"Window half-width (eV): {WINDOW_HALF_WIDTH_EV}")
    lines.append(f"Eta values: {etas}")
    lines.append(f"V0 values: {v0s}")
    lines.append(f"N_w values: {nws}")
    lines.append("")
    lines.append("Interpretation notes")
    lines.append("--------------------")
    lines.append(
        "Point 1: finite-window integrals are computed around diag_minus, "
        "diag_plus, cross_plus_minus, and cross_minus_plus using the hybridized "
        "bright-dark eigenenergies E_minus/E_plus."
    )
    lines.append(
        "Point 2: background subtraction first uses the V0=0 run at the same Eta, "
        "same N_w, and same named window. If that baseline is absent, the closest "
        "available N_w baseline at the same Eta/window is used and flagged."
    )
    lines.append(
        "Point 3: the Eta dependence should be read from the net cross-window "
        "metrics versus Eta. If apparent cross signal is dominated by linewidth "
        "leakage, it should decrease as Eta is reduced."
    )
    lines.append("")
    lines.append("Cross-window net abs_area")
    lines.append("-------------------------")
    for eta in etas:
        lines.append(f"Eta = {eta}")
        for row in sorted(cross_rows, key=lambda item: (item["V0"], item["N_w"], item["window"])):
            if row["eta"] == eta and row["V0"] != 0:
                lines.append(
                    f"  V0={row['V0']:.6g}, N_w={row['N_w']}, {row['window']}: "
                    f"abs_area={row['abs_area']:.8g}, "
                    f"abs_area_net={row['abs_area_net']:.8g}, "
                    f"max_abs={row['max_abs']:.8g}, "
                    f"max_abs_net={row['max_abs_net']:.8g}, "
                    f"background={row['background_mode']}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_eta_plots(rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (ANALYSIS_DIR / "eta_window_integrals_plot_skipped.txt").write_text(
            f"Matplotlib plot skipped: {exc!r}\n",
            encoding="utf-8",
        )
        return

    cross_rows = [
        row
        for row in rows
        if row["window"] == "cross_plus_minus"
        and row["V0"] != 0
        and row["N_w"] == 80
    ]
    if not cross_rows:
        return

    for metric, ylabel, filename in [
        ("abs_area_net", "Net finite-window |S| area", "eta_cross_abs_area_net.png"),
        ("max_abs_net", "Net finite-window max |S|", "eta_cross_max_abs_net.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for v0 in sorted({row["V0"] for row in cross_rows}):
            subset = sorted(
                [row for row in cross_rows if row["V0"] == v0],
                key=lambda item: item["eta"],
            )
            ax.plot(
                [row["eta"] for row in subset],
                [row[metric] for row in subset],
                marker="o",
                label=f"V0={v0:g}",
            )
        ax.axhline(0.0, color="0.4", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Eta (eV)")
        ax.set_ylabel(ylabel)
        ax.set_title("Step 1 cross-window background-subtracted signal")
        ax.legend()
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(ANALYSIS_DIR / filename, dpi=200)
        plt.close(fig)


def write_nw_plots(rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (ANALYSIS_DIR / "nw_window_integrals_plot_skipped.txt").write_text(
            f"Matplotlib plot skipped: {exc!r}\n",
            encoding="utf-8",
        )
        return

    cross_rows = [
        row
        for row in rows
        if row["window"] == "cross_plus_minus"
        and row["V0"] != 0
        and row["background_available"]
    ]
    groups = {}
    for row in cross_rows:
        groups.setdefault((row["eta"], row["V0"]), []).append(row)
    groups = {
        key: sorted(value, key=lambda item: item["N_w"])
        for key, value in groups.items()
        if len({item["N_w"] for item in value}) > 1
    }
    if not groups:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=False)
    for (eta, v0), subset in sorted(groups.items()):
        x = [row["N_w"] for row in subset]
        label = f"Eta={eta:g}, V0={v0:g}"
        axes[0].plot(x, [row["abs_area"] for row in subset], marker="o", label=label)
        axes[1].plot(x, [row["abs_area_net"] for row in subset], marker="o", label=label)

    axes[0].set_title("Raw cross-window |S| area")
    axes[1].set_title("Background-subtracted cross-window |S| area")
    for ax in axes:
        ax.set_xlabel("N_w")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Finite-window area")
    fig.suptitle("Step 1 N_w convergence check")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "nw_cross_abs_area_convergence.png", dpi=200)
    plt.close(fig)


def finite_number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def write_pdf_report(rows):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as exc:
        (ANALYSIS_DIR / "pdf_report_skipped.txt").write_text(
            f"PDF report skipped: {exc!r}\n",
            encoding="utf-8",
        )
        return None

    pdf_path = ANALYSIS_DIR / "step1_eta_nw_window_analysis.pdf"
    etas = sorted({row["eta"] for row in rows})
    v0s = sorted({row["V0"] for row in rows})
    nws = sorted({row["N_w"] for row in rows})
    data_files = sorted({row["file"] for row in rows})
    cross_rows = [
        row
        for row in rows
        if row["window"] == "cross_plus_minus"
        and row["V0"] != 0
        and row["background_available"]
    ]
    fallback_count = sum(
        1 for row in cross_rows if row["background_mode"] == "same_eta_closest_N_w"
    )

    eta_rows = [row for row in cross_rows if row["N_w"] == 80]
    nw_groups = {}
    for row in cross_rows:
        nw_groups.setdefault((row["eta"], row["V0"]), []).append(row)
    nw_groups = {
        key: sorted(value, key=lambda item: item["N_w"])
        for key, value in nw_groups.items()
        if len({item["N_w"] for item in value}) > 1
    }

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.94, "Step 1 - Analyse Eta et N_w", fontsize=18, weight="bold")
        lines = [
            f"Dossier: {RESULT_ROOT}",
            f"Fichiers S_data analyses: {len(data_files)}",
            f"Composante: {DEFAULT_COMPONENT}",
            f"Fenetre d'integration: +/- {WINDOW_HALF_WIDTH_EV:g} eV",
            f"Eta: {etas}",
            f"V0: {v0s}",
            f"N_w: {nws}",
            "",
            "Lecture principale:",
            "- Les scans Eta de reference sont lus sur N_w=80 pour ne pas les melanger avec les scans de grille.",
            "- Les scans N_w testent surtout la stabilite numerique des integrales de fenetre.",
            "- Une soustraction de fond utilise V0=0 au meme Eta/N_w si possible; sinon le N_w disponible le plus proche.",
            f"- Points cross-window avec baseline N_w approximative: {fallback_count}.",
        ]
        y = 0.88
        for line in lines:
            wrapped_lines = textwrap.wrap(line, width=92) if line else [""]
            for wrapped_line in wrapped_lines:
                fig.text(0.08, y, wrapped_line, fontsize=10)
                y -= 0.026
            y -= 0.006
        pdf.savefig(fig)
        plt.close(fig)

        if eta_rows:
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            for v0 in sorted({row["V0"] for row in eta_rows}):
                subset = sorted(
                    [row for row in eta_rows if row["V0"] == v0],
                    key=lambda item: item["eta"],
                )
                ax.plot(
                    [row["eta"] for row in subset],
                    [row["abs_area_net"] for row in subset],
                    marker="o",
                    label=f"V0={v0:g}",
                )
            ax.axhline(0.0, color="0.4", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_xlabel("Eta (eV)")
            ax.set_ylabel("Net cross-window |S| area")
            ax.set_title("Tendance Eta sur la grille de reference N_w=80")
            ax.legend()
            ax.grid(True, which="both", alpha=0.25)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        if nw_groups:
            fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=False)
            for (eta, v0), subset in sorted(nw_groups.items()):
                x = [row["N_w"] for row in subset]
                label = f"Eta={eta:g}, V0={v0:g}"
                axes[0].plot(
                    x,
                    [finite_number(row["abs_area"]) for row in subset],
                    marker="o",
                    label=label,
                )
                axes[1].plot(
                    x,
                    [finite_number(row["abs_area_net"]) for row in subset],
                    marker="o",
                    label=label,
                )
            axes[0].set_title("Signal brut")
            axes[1].set_title("Signal net")
            for ax in axes:
                ax.set_xlabel("N_w")
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=8)
            axes[0].set_ylabel("Cross-window |S| area")
            fig.suptitle("Convergence N_w - fenetre cross_plus_minus")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        table_rows = sorted(
            cross_rows,
            key=lambda item: (item["eta"], item["V0"], item["N_w"]),
        )
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.94, "Table courte - cross_plus_minus", fontsize=15, weight="bold")
        header = "Eta        V0      N_w    abs_area_net      max_abs_net      baseline"
        fig.text(0.08, 0.90, header, fontsize=8.5, family="monospace")
        y = 0.875
        for row in table_rows[:34]:
            line = (
                f"{row['eta']:<10.4g} {row['V0']:<7.3g} {row['N_w']:<6d} "
                f"{finite_number(row['abs_area_net']):<16.6g} "
                f"{finite_number(row['max_abs_net']):<16.6g} "
                f"{row['background_mode']}"
            )
            fig.text(0.08, y, line, fontsize=8.5, family="monospace")
            y -= 0.024
        if len(table_rows) > 34:
            fig.text(0.08, y, f"... {len(table_rows) - 34} lignes restantes dans le CSV.", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)

    return pdf_path


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    data_files = sorted(RESULT_ROOT.glob("*_S_data.npz"))
    rows = []
    skipped = []
    for data_file in data_files:
        try:
            rows.extend(analyze_file(data_file))
        except Exception as exc:
            skipped.append((data_file.name, repr(exc)))

    rows = add_background_subtraction(rows)
    write_csv(ANALYSIS_DIR / "eta_window_integrals.csv", rows)
    write_summary(ANALYSIS_DIR / "eta_window_integrals_summary.txt", rows)
    write_eta_plots(rows)
    write_nw_plots(rows)
    pdf_path = write_pdf_report(rows)

    if skipped:
        skipped_path = ANALYSIS_DIR / "eta_window_integrals_skipped.txt"
        skipped_path.write_text(
            "\n".join(f"{name}: {error}" for name, error in skipped) + "\n",
            encoding="utf-8",
        )
    else:
        skipped_path = ANALYSIS_DIR / "eta_window_integrals_skipped.txt"
        if skipped_path.exists():
            try:
                skipped_path.unlink()
            except PermissionError:
                pass

    print(f"Analyzed files: {len(data_files) - len(skipped)}")
    print(f"Skipped files: {len(skipped)}")
    print(f"CSV: {ANALYSIS_DIR / 'eta_window_integrals.csv'}")
    print(f"Summary: {ANALYSIS_DIR / 'eta_window_integrals_summary.txt'}")
    if pdf_path is not None:
        print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
