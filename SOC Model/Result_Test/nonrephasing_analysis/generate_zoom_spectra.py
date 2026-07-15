from pathlib import Path
import json
import os
import re
import subprocess
import sys

import numpy as np


ANALYSIS_DIR = Path(__file__).resolve().parent
RESULT_ROOT = ANALYSIS_DIR.parent
ZOOM_ROOT = ANALYSIS_DIR / "ZoomData"

ZOOM_N = 161
ZOOM_HALF_WIDTH_EV = 0.040

STEP_CONFIG = {
    "etape_6": {
        "script": RESULT_ROOT / "etape_6" / "step6_dynamic_phonon_parameter_scans.py",
        "env": "STEP6_ACTIVE_CASE",
        "cases": ["gQ_0", "gQ_0p01", "omegaQ_0p07"],
    },
    "etape_3": {
        "script": RESULT_ROOT / "etape_3" / "step3_dynamic_dimerisation_phonon.py",
        "env": "STEP3_ACTIVE_CASE",
        "cases": ["static_dimerisation", "dynamic_dimerisation_phonon"],
    },
    "etape_5": {
        "script": RESULT_ROOT / "etape_5" / "step5_temperature_field_scans.py",
        "env": "STEP5_ACTIVE_CASE",
        "cases": ["T_lattice_T_7", "T_lattice_T_20", "T_C1_static_T_20"],
    },
    "etape_2": {
        "script": RESULT_ROOT / "etape_2" / "step2_separate_soc_phonon_dimerization.py",
        "env": "STEP2_ACTIVE_CASE",
        "cases": ["dimerisation_control", "spin_correlation_control"],
    },
}


def summary_for_case(step, case):
    step_dir = RESULT_ROOT / step
    for summary_path in sorted((step_dir / "Summaries").glob("*_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("active_case_key") == case:
            return summary
    raise FileNotFoundError(f"No summary found for {step} {case}")


def e_plus_from_summary(summary):
    derived = summary.get("derived", {})
    if "E_plus" in derived:
        return float(derived["E_plus"])

    model = summary["model_parameters"]
    if "V_BD_eff" in derived:
        mixing = float(derived["V_BD_eff"])
    else:
        mixing = float(derived.get("V_BD_static", 0.0))
    delta_dark = float(model["Delta_dark"])
    delta_bright = float(model["Delta_Bright"])
    center = 0.5 * (delta_dark + delta_bright)
    split = np.sqrt((0.5 * (delta_bright - delta_dark)) ** 2 + mixing**2)
    return float(center + split)


def rewrite_source(source, step):
    replacement_root = 'RESULT_ROOT = Path(os.environ["ZOOM_RESULT_ROOT"])'
    source = re.sub(
        rf'RESULT_ROOT = PROJECT_ROOT / "SOC Model" / "Result_Test" / "{step}"',
        replacement_root,
        source,
    )
    axis_block = (
        "N_w = int(os.environ.get(\"ZOOM_N\", \"161\"))\n"
        "zoom_center = float(os.environ[\"ZOOM_CENTER_EV\"])\n"
        "zoom_half_width = float(os.environ.get(\"ZOOM_HALF_WIDTH_EV\", \"0.04\"))\n"
        "omega1_rephasing = np.linspace(-zoom_center - zoom_half_width, -zoom_center + zoom_half_width, N_w)\n"
        "omega1_unrephasing = np.linspace(zoom_center - zoom_half_width, zoom_center + zoom_half_width, N_w)\n"
        "omega1 = np.concatenate([omega1_rephasing, omega1_unrephasing])\n"
        "omega3 = np.linspace(zoom_center - zoom_half_width, zoom_center + zoom_half_width, N_w)"
    )
    source = re.sub(
        r"N_w = \d+\nomega1 = np\.linspace\([^\n]+\)\nomega3 = np\.linspace\([^\n]+\)",
        axis_block,
        source,
    )
    source = re.sub(
        r"(figure_directory = [^\n]+\n)",
        r"\1figure_directory.mkdir(parents=True, exist_ok=True)\n",
        source,
        count=1,
    )
    source = source.replace("save_pdf = True", "save_pdf = False", 1)
    source = source.replace("if save_pdf:\n    spectrum_data = {", "if True:\n    spectrum_data = {", 1)
    source = source.replace("show=False,", "show=False,")
    return source


def run_zoom_case(step, config, case):
    summary = summary_for_case(step, case)
    center = e_plus_from_summary(summary)
    source = config["script"].read_text(encoding="utf-8")
    rewritten = rewrite_source(source, step)

    runner = ANALYSIS_DIR / f"_zoom_runner_{step}_{case}.py"
    runner.write_text(rewritten, encoding="utf-8")

    zoom_result_root = ZOOM_ROOT / step / case
    if list((zoom_result_root / "Data").glob("*_S_data.npz")):
        return zoom_result_root

    env = os.environ.copy()
    env[config["env"]] = case
    env["ZOOM_RESULT_ROOT"] = str(zoom_result_root)
    env["ZOOM_CENTER_EV"] = f"{center:.12g}"
    env["ZOOM_HALF_WIDTH_EV"] = f"{ZOOM_HALF_WIDTH_EV:.12g}"
    env["ZOOM_N"] = str(ZOOM_N)

    subprocess.run(
        [sys.executable, "-B", str(runner)],
        cwd=RESULT_ROOT.parents[1],
        env=env,
        check=True,
    )
    return zoom_result_root


def main():
    ZOOM_ROOT.mkdir(parents=True, exist_ok=True)
    completed = []
    for step, config in STEP_CONFIG.items():
        for case in config["cases"]:
            root = run_zoom_case(step, config, case)
            completed.append((step, case, root))
            print(f"zoom data: {step} {case} -> {root}")

    lines = [
        "Zoom spectra generated for document plotting.",
        f"Zoom half-width (eV): {ZOOM_HALF_WIDTH_EV}",
        f"Zoom N per branch: {ZOOM_N}",
        "",
    ]
    for step, case, root in completed:
        lines.append(f"{step} {case}: {root}")
    (ANALYSIS_DIR / "zoom_spectra_manifest.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
