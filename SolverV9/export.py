"""Spectrum observable extraction and result export helpers."""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from .models import SpectrumResult


def _axis_step(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 1.0
    return float(np.mean(np.abs(np.diff(values))))


def _sanitize_key(name):
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_") or "unnamed"


def _token(value):
    if isinstance(value, (np.integer, int)):
        text = str(int(value))
    elif isinstance(value, (np.floating, float)):
        text = f"{float(value):.6g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p").replace(" ", "")


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _require_two_frequency_result(result):
    if not isinstance(result, SpectrumResult):
        raise TypeError("result must be a SpectrumResult.")
    if len(result.axis_names) != 2 or len(result.axis_values) != 2:
        raise ValueError("A two-frequency SpectrumResult is required.")
    y_values = np.asarray(result.axis_values[0], dtype=float)
    x_values = np.asarray(result.axis_values[1], dtype=float)
    return y_values, x_values


def _selected_names(source, names):
    if names is None or names == "all":
        selected = tuple(source)
    elif isinstance(names, str):
        selected = (names,)
    else:
        selected = tuple(str(name) for name in names)
    missing = [name for name in selected if name not in source]
    if missing:
        raise KeyError(f"Unknown spectrum name(s) {missing}; available={tuple(source)}.")
    if not selected:
        raise ValueError("At least one spectrum name must be selected.")
    return selected


def _select_spectra(result, spectra, names):
    source_key = str(spectra).lower() if isinstance(spectra, str) else None
    if source_key == "pathways":
        source = result.pathways
    elif source_key == "components":
        source = result.components
    elif hasattr(spectra, "items"):
        source = {str(name): np.asarray(values) for name, values in spectra.items()}
    else:
        raise ValueError("spectra must be 'components', 'pathways', or a mapping.")

    selected = _selected_names(source, names)
    return {name: np.asarray(source[name]) for name in selected}


def _axis_bounds_mask(values, bounds):
    values = np.asarray(values, dtype=float)
    lo, hi = sorted(float(value) for value in bounds)
    mask = (values >= lo) & (values <= hi)
    if not np.any(mask):
        raise ValueError(f"No axis samples fall inside bounds ({lo}, {hi}).")
    return mask


def _quantity_values(data, quantity):
    key = str(quantity).lower()
    if key in ("abs", "absolute", "magnitude"):
        return np.abs(data)
    if key in ("real", "re"):
        return np.real(data)
    if key in ("imag", "imaginary", "im"):
        return np.imag(data)
    raise ValueError("quantity must be 'abs', 'real', or 'imag'.")


def _axis_window_from_frequency(frequency, width):
    if frequency is None:
        raise ValueError("frequency is required when no window is provided.")
    if width is None:
        raise ValueError("width is required when frequency is used.")
    half_width = 0.5 * float(width)
    center = float(frequency)
    return center - half_width, center + half_width


def _center_coordinates(center, axis_names):
    if center is None:
        raise ValueError("center is required for local diagonal/off-diagonal cuts.")
    if not isinstance(center, dict):
        try:
            y_center, x_center = center
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "center must be a mapping with omega1/omega3 values or "
                "a pair (omega1_center, omega3_center)."
            ) from exc
        return float(y_center), float(x_center)

    y_aliases = (
        str(axis_names[0]),
        "omega1",
        "omega_1",
        "omega_1Q",
        "w1",
        "y",
    )
    x_aliases = (
        str(axis_names[1]),
        "omega3",
        "omega_3",
        "omega_emit",
        "w3",
        "x",
    )
    y_center = next((center[key] for key in y_aliases if key in center), None)
    x_center = next((center[key] for key in x_aliases if key in center), None)
    if y_center is None or x_center is None:
        raise ValueError(
            "center must define both frequency axes. "
            "Use keys such as 'omega1' and 'omega3'."
        )
    return float(y_center), float(x_center)


def _bilinear_interpolate_grid(y_values, x_values, values, y_points, x_points):
    """Sample a rectilinear 2D grid at continuous coordinates."""
    y_values = np.asarray(y_values, dtype=float)
    x_values = np.asarray(x_values, dtype=float)
    values = np.asarray(values)
    y_points = np.asarray(y_points, dtype=float)
    x_points = np.asarray(x_points, dtype=float)

    if y_values.size < 2 or x_values.size < 2:
        raise ValueError("Interpolation requires at least two samples on each axis.")
    if np.any(np.diff(y_values) <= 0.0) or np.any(np.diff(x_values) <= 0.0):
        raise ValueError("Interpolation requires strictly increasing frequency axes.")

    flat_y = y_points.ravel()
    flat_x = x_points.ravel()
    valid = (
        (flat_y >= y_values[0])
        & (flat_y <= y_values[-1])
        & (flat_x >= x_values[0])
        & (flat_x <= x_values[-1])
    )
    out = np.full(flat_y.shape, np.nan, dtype=float)
    if not np.any(valid):
        return out.reshape(y_points.shape)

    y_valid = flat_y[valid]
    x_valid = flat_x[valid]
    yi = np.searchsorted(y_values, y_valid, side="right") - 1
    xi = np.searchsorted(x_values, x_valid, side="right") - 1
    yi = np.clip(yi, 0, y_values.size - 2)
    xi = np.clip(xi, 0, x_values.size - 2)

    y0 = y_values[yi]
    y1 = y_values[yi + 1]
    x0 = x_values[xi]
    x1 = x_values[xi + 1]
    ty = (y_valid - y0) / (y1 - y0)
    tx = (x_valid - x0) / (x1 - x0)

    v00 = values[yi, xi]
    v01 = values[yi, xi + 1]
    v10 = values[yi + 1, xi]
    v11 = values[yi + 1, xi + 1]
    out[valid] = (
        (1.0 - ty) * (1.0 - tx) * v00
        + (1.0 - ty) * tx * v01
        + ty * (1.0 - tx) * v10
        + ty * tx * v11
    )
    return out.reshape(y_points.shape)


def _local_line_profile(
    values,
    y_values,
    x_values,
    *,
    center,
    half_length,
    width,
    cut,
    diagonal,
    num_points,
    axis_names,
):
    if half_length is None:
        raise ValueError("half_length is required for local line cuts.")
    if width is None:
        raise ValueError("width is required for local line cuts.")

    cut_key = str(cut).lower()
    diag_key = str(diagonal).lower()
    if diag_key in ("rephasing", "r"):
        if cut_key in ("diagonal", "diag"):
            direction = (1.0, -1.0)
            cut_name = "diagonal"
        elif cut_key in ("off_diagonal", "offdiag", "cross_diagonal"):
            direction = (1.0, 1.0)
            cut_name = "off_diagonal"
        else:
            raise ValueError("cut must be 'diagonal' or 'off_diagonal'.")
    elif diag_key in ("unrephasing", "ur", "nonrephasing"):
        if cut_key in ("diagonal", "diag"):
            direction = (1.0, 1.0)
            cut_name = "diagonal"
        elif cut_key in ("off_diagonal", "offdiag", "cross_diagonal"):
            direction = (1.0, -1.0)
            cut_name = "off_diagonal"
        else:
            raise ValueError("cut must be 'diagonal' or 'off_diagonal'.")
    else:
        raise ValueError("diagonal must be 'rephasing' or 'unrephasing'.")

    y_center, x_center = _center_coordinates(center, axis_names)
    direction_y, direction_x = direction

    half_length = float(half_length)
    width = float(width)
    n_points = int(num_points or max(x_values.size, y_values.size))
    profile_axis = np.linspace(-half_length, half_length, n_points)
    min_step = min(_axis_step(y_values), _axis_step(x_values))
    n_transverse = max(5, int(np.ceil(width / min_step)) * 4 + 1)
    if n_transverse % 2 == 0:
        n_transverse += 1
    transverse_axis = np.linspace(-0.5 * width, 0.5 * width, n_transverse)

    along_grid = profile_axis[:, None]
    transverse_grid = transverse_axis[None, :]
    sample_y = y_center + direction_y * (along_grid + transverse_grid)
    sample_x = x_center + direction_x * (along_grid - transverse_grid)
    sampled = _bilinear_interpolate_grid(
        y_values,
        x_values,
        values,
        sample_y,
        sample_x,
    )
    if np.all(np.isnan(sampled)):
        raise ValueError("No samples fall inside the requested local line cut.")
    # Average over the transverse strip without changing the spectrum units.
    # The width smooths the slice but should not rescale the peak amplitude.
    valid_counts = np.sum(np.isfinite(sampled), axis=1)
    profile_values = np.divide(
        np.nansum(sampled, axis=1),
        valid_counts,
        out=np.full(profile_axis.shape, np.nan, dtype=float),
        where=valid_counts > 0,
    )
    omega1_line = y_center + direction_y * profile_axis
    omega3_line = x_center + direction_x * profile_axis

    excitation_sign = -1.0 if diag_key in ("rephasing", "r") else 1.0
    if cut_name == "diagonal":
        energy_axis = 0.5 * (excitation_sign * omega1_line + omega3_line)
        axis_name = "diagonal_energy_eV"
    else:
        energy_axis = omega3_line
        axis_name = f"{axis_names[1]}_eV"
    order = np.argsort(energy_axis)

    return {
        "axis": energy_axis[order],
        "intensity": profile_values[order],
        "axis_name": axis_name,
        "cut": cut_name,
        "integrated_axis": "transverse_width",
        "diagonal": diag_key,
        "center": {
            str(axis_names[0]): y_center,
            str(axis_names[1]): x_center,
        },
        "half_length": half_length,
        "width": width,
        "center_energy_eV": float(
            0.5 * (excitation_sign * y_center + x_center)
        ),
        "detuning_from_center": profile_axis[order],
        "omega1_line": omega1_line[order],
        "omega3_line": omega3_line[order],
    }


def _window_bounds(window, axis_names):
    if window is None:
        raise ValueError(
            "cross_window and diag_window must be provided to compute "
            "Icross, Idiag, and Rcross."
        )
    if isinstance(window, dict):
        y_aliases = (
            str(axis_names[0]),
            "omega1",
            "omega_1",
            "omega_1Q",
            "w1",
            "y",
        )
        x_aliases = (
            str(axis_names[1]),
            "omega3",
            "omega_3",
            "omega_emit",
            "w3",
            "x",
        )
        y_bounds = next((window[key] for key in y_aliases if key in window), None)
        x_bounds = next((window[key] for key in x_aliases if key in window), None)
    else:
        try:
            y_bounds, x_bounds = window
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "A window must be a mapping with omega1/omega3 bounds or "
                "a pair of (omega1_bounds, omega3_bounds)."
            ) from exc
    if y_bounds is None or x_bounds is None:
        raise ValueError(
            "Each window must define bounds for both frequency axes. "
            "Use keys such as 'omega1' and 'omega3'."
        )
    if len(y_bounds) != 2 or len(x_bounds) != 2:
        raise ValueError("Window bounds must contain exactly two values per axis.")
    y_lo, y_hi = sorted(float(value) for value in y_bounds)
    x_lo, x_hi = sorted(float(value) for value in x_bounds)
    return (y_lo, y_hi), (x_lo, x_hi)


def _window_mask(window, y_values, x_values, axis_names):
    (y_lo, y_hi), (x_lo, x_hi) = _window_bounds(window, axis_names)
    y_mask = (y_values >= y_lo) & (y_values <= y_hi)
    x_mask = (x_values >= x_lo) & (x_values <= x_hi)
    if not np.any(y_mask) or not np.any(x_mask):
        raise ValueError(
            "No grid samples fall inside the requested window "
            f"omega1=({y_lo}, {y_hi}), omega3=({x_lo}, {x_hi})."
        )
    return np.outer(y_mask, x_mask), {
        str(axis_names[0]): (y_lo, y_hi),
        str(axis_names[1]): (x_lo, x_hi),
    }


def _profile_scalar_observables(profile, prefix):
    axis = np.asarray(profile["axis"], dtype=float)
    intensity = np.asarray(profile["intensity"])
    if axis.size == 0 or intensity.size == 0:
        raise ValueError(f"Profile {prefix!r} is empty.")
    abs_intensity = np.abs(intensity)
    finite = np.isfinite(abs_intensity)
    if not np.any(finite):
        raise ValueError(f"Profile {prefix!r} has no finite intensity values.")
    finite_abs = np.where(finite, abs_intensity, -np.inf)
    max_index = int(np.argmax(finite_abs))
    values = {
        f"{prefix}_Iint": float(np.nansum(intensity)),
        f"{prefix}_Iabsint": float(np.nansum(abs_intensity)),
        f"{prefix}_Imax": float(abs_intensity[max_index]),
        f"{prefix}_coord_star": float(axis[max_index]),
    }
    if "omega1_line" in profile:
        omega1_line = np.asarray(profile["omega1_line"], dtype=float)
        values[f"{prefix}_omega1_star"] = float(omega1_line[max_index])
    if "omega3_line" in profile:
        omega3_line = np.asarray(profile["omega3_line"], dtype=float)
        values[f"{prefix}_omega3_star"] = float(omega3_line[max_index])
    if "center_energy_eV" in profile:
        values[f"{prefix}_center_energy_eV"] = float(profile["center_energy_eV"])
    return values


def _feature_half_widths(
    feature_centers,
    axis_names,
    half_width,
    *,
    fraction,
    fallback,
):
    if isinstance(half_width, str) and half_width.lower() == "auto":
        coordinates = [
            _center_coordinates(center, axis_names)
            for center in feature_centers.values()
        ]
        half_widths = []
        for axis_index in (0, 1):
            unique = sorted({round(float(coord[axis_index]), 12) for coord in coordinates})
            spacings = [
                b - a
                for a, b in zip(unique[:-1], unique[1:])
                if b > a
            ]
            half_widths.append(float(fraction * min(spacings)) if spacings else float(fallback))
        return tuple(half_widths)
    if isinstance(half_width, dict):
        return (
            float(half_width.get(axis_names[0], half_width.get("omega1"))),
            float(half_width.get(axis_names[1], half_width.get("omega3"))),
        )
    value = float(half_width)
    return value, value


def _window_from_center(center, half_widths, axis_names):
    y_center, x_center = _center_coordinates(center, axis_names)
    y_half_width, x_half_width = half_widths
    return {
        str(axis_names[0]): (y_center - y_half_width, y_center + y_half_width),
        str(axis_names[1]): (x_center - x_half_width, x_center + x_half_width),
    }


def _find_feature_key(feature_mapping, *candidates):
    by_sanitized = {
        _sanitize_key(name).lower(): name
        for name in feature_mapping
    }
    for candidate in candidates:
        key = by_sanitized.get(_sanitize_key(candidate).lower())
        if key is not None:
            return key
    return None


def find_spectrum_feature_centers(
    result,
    *,
    feature_centers,
    spectra="pathways",
    names=("R1", "R2"),
    detection_phase=0.0,
    quantity="abs",
    search_half_width="auto",
    search_fraction=0.2,
    fallback_search_half_width=0.03,
):
    """Refine nominal feature centers by local peak search in a 2D spectrum."""
    if not feature_centers:
        raise ValueError("feature_centers must provide at least one nominal center.")

    y_values, x_values = _require_two_frequency_result(result)
    selected = _select_spectra(result, spectra, names)
    expected_shape = (y_values.size, x_values.size)
    for name, values in selected.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"Spectrum {name!r} has shape {values.shape}; expected {expected_shape}."
            )

    phase = np.exp(1j * (0.0 if detection_phase is None else float(detection_phase)))
    total = sum(phase * values for values in selected.values())
    search_values = np.abs(_quantity_values(total, quantity))
    half_widths = _feature_half_widths(
        feature_centers,
        result.axis_names,
        search_half_width,
        fraction=float(search_fraction),
        fallback=float(fallback_search_half_width),
    )

    centers = {}
    for feature_name, nominal_center in feature_centers.items():
        search_window = _window_from_center(
            nominal_center,
            half_widths,
            result.axis_names,
        )
        mask, bounds = _window_mask(
            search_window,
            y_values,
            x_values,
            result.axis_names,
        )
        if not np.any(mask):
            raise ValueError(f"No samples fall inside search window for {feature_name!r}.")
        y_indices, x_indices = np.where(mask)
        local_values = search_values[mask]
        peak = int(np.argmax(local_values))
        iy = int(y_indices[peak])
        ix = int(x_indices[peak])
        centers[feature_name] = {
            "center": {
                str(result.axis_names[0]): float(y_values[iy]),
                str(result.axis_names[1]): float(x_values[ix]),
            },
            "nominal_center": {
                str(result.axis_names[0]): float(_center_coordinates(nominal_center, result.axis_names)[0]),
                str(result.axis_names[1]): float(_center_coordinates(nominal_center, result.axis_names)[1]),
            },
            "search_window": bounds,
            "Imax": float(search_values[iy, ix]),
        }
    return centers


def extract_spectrum_observables(
    result,
    *,
    spectra="pathways",
    names=("R1", "R2"),
    detection_phase=0.0,
    cross_window=None,
    diag_window=None,
    feature_centers=None,
    auto_center_features=True,
    feature_search_half_width="auto",
    feature_search_fraction=0.2,
    feature_window_half_width="auto",
    feature_window_fraction=0.25,
    feature_specs=None,
    feature_quantity="abs",
    feature_cuts=("diagonal", "off_diagonal"),
):
    """Return scalar observables from an unnormalized two-dimensional spectrum.

    ``cross_window`` and ``diag_window`` may define bounds for both axes, for
    example ``{"omega1": (-1.08, -1.04), "omega3": (1.04, 1.09)}``.
    Alternatively, ``feature_centers`` may provide nominal centers named for
    example ``Bright``, ``Cross_down``, ``Cross_up``, and ``Dark``.  In that
    case the centers are refined by a local maximum search, and integration
    windows are built automatically from the feature spacing.

    ``feature_specs`` may define local feature centers for diagonal and
    off-diagonal cut observables.  Each entry must provide ``center``,
    ``half_length``, and ``width``.
    """
    y_values, x_values = _require_two_frequency_result(result)
    selected = _select_spectra(result, spectra, names)
    expected_shape = (y_values.size, x_values.size)
    for name, values in selected.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"Spectrum {name!r} has shape {values.shape}; expected {expected_shape}."
            )

    phase = np.exp(1j * (0.0 if detection_phase is None else float(detection_phase)))
    selected = {name: phase * values for name, values in selected.items()}
    total = sum(selected.values())
    magnitude = np.abs(total)
    cell_area = _axis_step(y_values) * _axis_step(x_values)

    found_features = None
    feature_windows = {}
    if feature_centers:
        if auto_center_features:
            found_features = find_spectrum_feature_centers(
                result,
                feature_centers=feature_centers,
                spectra=spectra,
                names=names,
                detection_phase=detection_phase,
                quantity=feature_quantity,
                search_half_width=feature_search_half_width,
                search_fraction=feature_search_fraction,
            )
        else:
            found_features = {
                name: {
                    "center": {
                        str(result.axis_names[0]): float(_center_coordinates(center, result.axis_names)[0]),
                        str(result.axis_names[1]): float(_center_coordinates(center, result.axis_names)[1]),
                    },
                    "nominal_center": center,
                    "search_window": None,
                    "Imax": math.nan,
                }
                for name, center in feature_centers.items()
            }
        window_half_widths = _feature_half_widths(
            feature_centers,
            result.axis_names,
            feature_window_half_width,
            fraction=float(feature_window_fraction),
            fallback=0.025,
        )
        for feature_name, info in found_features.items():
            feature_windows[feature_name] = _window_from_center(
                info["center"],
                window_half_widths,
                result.axis_names,
            )

        cross_key = _find_feature_key(feature_windows, "Cross_down", "Cross")
        diag_key = _find_feature_key(feature_windows, "Bright", "Diag", "Diagonal")
        if cross_window is None and cross_key is not None:
            cross_window = feature_windows[cross_key]
        if diag_window is None and diag_key is not None:
            diag_window = feature_windows[diag_key]

    if cross_window is None or diag_window is None:
        raise ValueError(
            "Provide cross_window and diag_window, or provide feature_centers "
            "containing at least Bright and Cross_down."
        )

    max_index = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    cross_mask, cross_bounds = _window_mask(
        cross_window,
        y_values,
        x_values,
        result.axis_names,
    )
    diag_mask, diag_bounds = _window_mask(
        diag_window,
        y_values,
        x_values,
        result.axis_names,
    )

    i_cross = float(np.sum(magnitude[cross_mask]) * cell_area)
    i_diag = float(np.sum(magnitude[diag_mask]) * cell_area)
    observables = {
        "Imax": float(magnitude[max_index]),
        "Iint": float(np.sum(magnitude) * cell_area),
        "Icross": i_cross,
        "Idiag": i_diag,
        "Rcross": float(i_cross / i_diag) if i_diag != 0 else math.nan,
        "IR1": float(np.sum(np.abs(selected["R1"])) * cell_area)
        if "R1" in selected else math.nan,
        "IR2": float(np.sum(np.abs(selected["R2"])) * cell_area)
        if "R2" in selected else math.nan,
        "omega1_star": float(y_values[max_index[0]]),
        "omega3_star": float(x_values[max_index[1]]),
        "cross_window": cross_bounds,
        "diag_window": diag_bounds,
    }

    if found_features:
        observables["feature_centers"] = {
            name: info["center"]
            for name, info in found_features.items()
        }
        observables["feature_search_windows"] = {
            name: info["search_window"]
            for name, info in found_features.items()
        }
        observables["feature_windows"] = feature_windows
        for feature_name, window in feature_windows.items():
            feature_key = _sanitize_key(feature_name)
            feature_mask, feature_bounds = _window_mask(
                window,
                y_values,
                x_values,
                result.axis_names,
            )
            if np.any(feature_mask):
                local_magnitude = magnitude[feature_mask]
                y_indices, x_indices = np.where(feature_mask)
                peak = int(np.argmax(local_magnitude))
                iy = int(y_indices[peak])
                ix = int(x_indices[peak])
                observables[f"{feature_key}_Iint"] = float(np.sum(local_magnitude) * cell_area)
                observables[f"{feature_key}_Imax"] = float(local_magnitude[peak])
                observables[f"{feature_key}_omega1_star"] = float(y_values[iy])
                observables[f"{feature_key}_omega3_star"] = float(x_values[ix])
            else:
                observables[f"{feature_key}_Iint"] = math.nan
                observables[f"{feature_key}_Imax"] = math.nan
                observables[f"{feature_key}_omega1_star"] = math.nan
                observables[f"{feature_key}_omega3_star"] = math.nan
            observables[f"{feature_key}_window"] = feature_bounds

        bright_key = _find_feature_key(feature_windows, "Bright")
        dark_key = _find_feature_key(feature_windows, "Dark")
        cross_down_key = _find_feature_key(feature_windows, "Cross_down")
        cross_up_key = _find_feature_key(feature_windows, "Cross_up")
        if bright_key is not None and cross_down_key is not None:
            bright_iint = observables.get(f"{_sanitize_key(bright_key)}_Iint", math.nan)
            cross_down_iint = observables.get(f"{_sanitize_key(cross_down_key)}_Iint", math.nan)
            observables["Rcross_down"] = (
                float(cross_down_iint / bright_iint) if bright_iint else math.nan
            )
        if bright_key is not None and cross_up_key is not None:
            bright_iint = observables.get(f"{_sanitize_key(bright_key)}_Iint", math.nan)
            cross_up_iint = observables.get(f"{_sanitize_key(cross_up_key)}_Iint", math.nan)
            observables["Rcross_up"] = (
                float(cross_up_iint / bright_iint) if bright_iint else math.nan
            )
        cross_total = 0.0
        cross_count = 0
        for key in (cross_down_key, cross_up_key):
            if key is not None:
                value = observables.get(f"{_sanitize_key(key)}_Iint", math.nan)
                if not math.isnan(value):
                    cross_total += value
                    cross_count += 1
        diag_total = 0.0
        diag_count = 0
        for key in (bright_key, dark_key):
            if key is not None:
                value = observables.get(f"{_sanitize_key(key)}_Iint", math.nan)
                if not math.isnan(value):
                    diag_total += value
                    diag_count += 1
        if cross_count:
            observables["Icross_total"] = float(cross_total)
        if diag_count:
            observables["Idiag_total"] = float(diag_total)
        if cross_count and diag_count and diag_total:
            observables["Rcross_total"] = float(cross_total / diag_total)

    if feature_specs:
        for feature_name, spec in feature_specs.items():
            feature_key = _sanitize_key(feature_name)
            center = spec["center"]
            if found_features and feature_name in found_features:
                center = found_features[feature_name]["center"]
            cuts = spec.get("cuts", feature_cuts)
            if isinstance(cuts, str):
                cuts = (cuts,)
            for cut in cuts:
                cut_key = _sanitize_key(cut)
                profile = extract_spectrum_profile(
                    result,
                    spectra=spectra,
                    names=names,
                    detection_phase=detection_phase,
                    quantity=spec.get("quantity", feature_quantity),
                    cut=cut,
                    center=center,
                    half_length=spec["half_length"],
                    width=spec["width"],
                    quadrant=spec.get("quadrant", spec.get("diagonal", "rephasing")),
                    num_points=spec.get("num_points"),
                )
                prefix = f"{feature_key}_{cut_key}"
                observables.update(_profile_scalar_observables(profile, prefix))
                observables[f"{prefix}_center"] = profile.get("center")
                observables[f"{prefix}_half_length"] = profile.get("half_length")
                observables[f"{prefix}_width"] = profile.get("width")
    observables["feature_quantity"] = feature_quantity
    return observables


def extract_spectrum_profile(
    result,
    *,
    spectra="pathways",
    names=("R1", "R2"),
    detection_phase=0.0,
    quantity="abs",
    cut="omega3",
    window=None,
    center=None,
    half_length=None,
    frequency=None,
    width=None,
    diagonal="rephasing",
    quadrant=None,
    offset=0.0,
    num_points=None,
):
    """Extract a one-dimensional profile from a two-dimensional spectrum.

    Parameters
    ----------
    cut : {"omega3", "omega1", "diagonal", "off_diagonal"}
        ``omega3`` keeps the emission axis and integrates over omega1.
        ``omega1`` keeps the coherence axis and integrates over omega3.
        With ``center``, ``diagonal`` and ``off_diagonal`` are local line cuts
        through the requested feature center.
    window : mapping or None
        Optional 2D window.  For ``omega1`` and ``omega3`` cuts, the transverse
        part of the window is integrated and the kept axis is restricted to the
        window bounds.  If omitted, ``frequency`` and ``width`` define the
        transverse slice.
    center : mapping, pair, or None
        Center of a local feature, for example
        ``{"omega1": -1.06, "omega3": 1.06}``.
    half_length : float or None
        Half-length of a local cut around ``center``.
    frequency : float or None
        Required for ``off_diagonal`` and for axis cuts when ``window`` is not
        supplied.
    width : float or None
        Width of the integrated slice.  Required when ``frequency`` is used.
    diagonal : {"rephasing", "unrephasing"}
        Frequency convention used to convert omega1 into a positive excitation
        frequency before diagonal/off-diagonal cuts.
    quadrant : {"rephasing", "unrephasing"} or None
        Alias for ``diagonal``.  When supplied, it takes precedence.
    """
    y_values, x_values = _require_two_frequency_result(result)
    selected = _select_spectra(result, spectra, names)
    expected_shape = (y_values.size, x_values.size)
    for name, values in selected.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"Spectrum {name!r} has shape {values.shape}; expected {expected_shape}."
            )

    phase = np.exp(1j * (0.0 if detection_phase is None else float(detection_phase)))
    total = sum(phase * values for values in selected.values())
    values = _quantity_values(total, quantity)
    y_step = _axis_step(y_values)
    x_step = _axis_step(x_values)
    cut_key = str(cut).lower()
    diagonal_key = diagonal if quadrant is None else quadrant

    if cut_key in ("omega3", "w3", "x"):
        if window is not None:
            y_bounds, x_bounds = _window_bounds(window, result.axis_names)
            y_mask = _axis_bounds_mask(y_values, y_bounds)
            x_mask = _axis_bounds_mask(x_values, x_bounds)
        elif center is not None:
            y_center, x_center = _center_coordinates(center, result.axis_names)
            if half_length is None:
                raise ValueError("half_length is required when center is used.")
            y_mask = _axis_bounds_mask(
                y_values,
                _axis_window_from_frequency(y_center, width),
            )
            x_mask = _axis_bounds_mask(
                x_values,
                (float(x_center) - float(half_length), float(x_center) + float(half_length)),
            )
        else:
            y_mask = _axis_bounds_mask(
                y_values,
                _axis_window_from_frequency(frequency, width),
            )
            x_mask = np.ones_like(x_values, dtype=bool)
        profile_axis = x_values[x_mask]
        profile_values = np.sum(values[np.ix_(y_mask, x_mask)], axis=0) * y_step
        return {
            "axis": profile_axis,
            "intensity": profile_values,
            "axis_name": f"{result.axis_names[1]}_eV",
            "quantity": str(quantity),
            "cut": "omega3",
            "integrated_axis": str(result.axis_names[0]),
        }

    if cut_key in ("omega1", "w1", "y"):
        if window is not None:
            y_bounds, x_bounds = _window_bounds(window, result.axis_names)
            y_mask = _axis_bounds_mask(y_values, y_bounds)
            x_mask = _axis_bounds_mask(x_values, x_bounds)
        elif center is not None:
            y_center, x_center = _center_coordinates(center, result.axis_names)
            if half_length is None:
                raise ValueError("half_length is required when center is used.")
            x_mask = _axis_bounds_mask(
                x_values,
                _axis_window_from_frequency(x_center, width),
            )
            y_mask = _axis_bounds_mask(
                y_values,
                (float(y_center) - float(half_length), float(y_center) + float(half_length)),
            )
        else:
            x_mask = _axis_bounds_mask(
                x_values,
                _axis_window_from_frequency(frequency, width),
            )
            y_mask = np.ones_like(y_values, dtype=bool)
        profile_axis = y_values[y_mask]
        profile_values = np.sum(values[np.ix_(y_mask, x_mask)], axis=1) * x_step
        return {
            "axis": profile_axis,
            "intensity": profile_values,
            "axis_name": f"{result.axis_names[0]}_eV",
            "quantity": str(quantity),
            "cut": "omega1",
            "integrated_axis": str(result.axis_names[1]),
        }

    if center is not None and cut_key in (
        "diagonal",
        "diag",
        "off_diagonal",
        "offdiag",
        "cross_diagonal",
    ):
        profile = _local_line_profile(
            values,
            y_values,
            x_values,
            center=center,
            half_length=half_length,
            width=width,
            cut=cut_key,
            diagonal=diagonal_key,
            num_points=num_points,
            axis_names=result.axis_names,
        )
        profile["quantity"] = str(quantity)
        return profile

    diag_key = str(diagonal_key).lower()
    if diag_key in ("rephasing", "r"):
        excitation_sign = -1.0
    elif diag_key in ("unrephasing", "ur", "nonrephasing"):
        excitation_sign = 1.0
    else:
        raise ValueError("diagonal must be 'rephasing' or 'unrephasing'.")

    excitation = excitation_sign * y_values[:, None]
    emission = x_values[None, :]
    diag_coord = 0.5 * (excitation + emission)
    anti_coord = emission - excitation
    cell_area = y_step * x_step

    if cut_key in ("diagonal", "diag"):
        if width is None:
            raise ValueError("width is required for a diagonal cut.")
        if window is not None:
            y_bounds, x_bounds = _window_bounds(window, result.axis_names)
            window_mask = np.outer(
                _axis_bounds_mask(y_values, y_bounds),
                _axis_bounds_mask(x_values, x_bounds),
            )
        else:
            window_mask = np.ones_like(values, dtype=bool)
        anti_mask = np.abs(anti_coord - float(offset)) <= 0.5 * float(width)
        masked = window_mask & anti_mask
        if not np.any(masked):
            raise ValueError("No samples fall inside the requested diagonal strip.")
        axis_min = float(np.min(diag_coord[masked]))
        axis_max = float(np.max(diag_coord[masked]))
        n_points = int(num_points or max(x_values.size, y_values.size))
        profile_axis = np.linspace(axis_min, axis_max, n_points)
        bin_width = (axis_max - axis_min) / max(n_points - 1, 1)
        profile_values = np.zeros_like(profile_axis)
        for index, center in enumerate(profile_axis):
            bin_mask = np.abs(diag_coord - center) <= 0.5 * bin_width
            profile_values[index] = np.sum(values[masked & bin_mask]) * cell_area
        return {
            "axis": profile_axis,
            "intensity": profile_values,
            "axis_name": "diagonal_frequency",
            "quantity": str(quantity),
            "cut": "diagonal",
            "integrated_axis": "anti_diagonal",
            "diagonal": diag_key,
            "offset": float(offset),
        }

    if cut_key in ("off_diagonal", "offdiag", "cross_diagonal"):
        if frequency is None:
            raise ValueError("frequency is required for an off-diagonal cut.")
        if width is None:
            raise ValueError("width is required for an off-diagonal cut.")
        diag_mask = np.abs(diag_coord - float(frequency)) <= 0.5 * float(width)
        if window is not None:
            y_bounds, x_bounds = _window_bounds(window, result.axis_names)
            diag_mask &= np.outer(
                _axis_bounds_mask(y_values, y_bounds),
                _axis_bounds_mask(x_values, x_bounds),
            )
        if not np.any(diag_mask):
            raise ValueError("No samples fall inside the requested off-diagonal strip.")
        axis_min = float(np.min(anti_coord[diag_mask]))
        axis_max = float(np.max(anti_coord[diag_mask]))
        n_points = int(num_points or max(x_values.size, y_values.size))
        profile_axis = np.linspace(axis_min, axis_max, n_points)
        bin_width = (axis_max - axis_min) / max(n_points - 1, 1)
        profile_values = np.zeros_like(profile_axis)
        for index, center in enumerate(profile_axis):
            bin_mask = np.abs(anti_coord - center) <= 0.5 * bin_width
            profile_values[index] = np.sum(values[diag_mask & bin_mask]) * cell_area
        return {
            "axis": profile_axis,
            "intensity": profile_values,
            "axis_name": "anti_diagonal_detuning",
            "quantity": str(quantity),
            "cut": "off_diagonal",
            "integrated_axis": "diagonal",
            "diagonal": diag_key,
            "frequency": float(frequency),
        }

    raise ValueError(
        "cut must be 'omega3', 'omega1', 'diagonal', or 'off_diagonal'."
    )


def make_run_id(
    *,
    ProjetID= 'SOC',
    scan_name=None,
    Scan_number=0,
    values=None,
):
    """Build a linked run identifier for data, figures, and analysis files."""
    pieces = [str(ProjetID), f"{scan_name}-{int(Scan_number):03d}"]
    if values:
        value_token = "_".join(f"{_sanitize_key(key)}-{_token(val)}" for key, val in values.items())
        pieces.append(value_token)
    return "__".join(pieces)


def build_mechanism_note(model_params, meta=None):
    """Return a short Markdown note listing active and inactive mechanisms."""
    meta = {} if meta is None else meta

    def active_scalar(name):
        return abs(float(model_params.get(name, 0.0))) > 0.0

    def active_list(name):
        return any(abs(float(value)) > 0.0 for value in model_params.get(name, []))

    entries = [
        ("Static bright-dark mixing V0", active_scalar("V0")),
        ("Dimerization contribution lambda_delta * delta", active_scalar("lambda_delta")),
        ("Spin-correlation contribution lambda_C * C1", active_scalar("lambda_C")),
        ("Direct dark dipole mu_D", active_scalar("mu_D")),
        ("Bright orbital hopping", active_list("bright_hoppings_eV")),
        ("Dark orbital hopping", active_list("dark_hoppings_eV")),
        ("Spin-mode dispersion", active_scalar("spin_mode_dispersion_scale")),
        ("Bright-dark k modulation", active_scalar("bright_dark_k_modulation")),
        ("Spin-phonon coupling g_Q", active_scalar("g_Q")),
        ("Spin-phonon k modulation", active_scalar("g_Q_k_modulation")),
        ("k integration", int(model_params.get("N_k", 1)) > 1),
    ]

    lines = ["# Mechanism note", ""]
    for label, is_active in entries:
        lines.append(f"- {label}: {'active' if is_active else 'inactive'}")
    if meta:
        lines.extend(
            [
                "",
                "## Derived quantities",
                f"- delta: {meta.get('delta', 'not available')}",
                f"- C1: {meta.get('C1', 'not available')}",
                f"- V_BD_static: {meta.get('V_BD_static', 'not available')}",
                f"- N_k: {len(meta.get('k_array', [])) if 'k_array' in meta else 'not available'}",
            ]
        )
    return "\n".join(lines) + "\n"


def _save_raw_spectra(path, result):
    arrays = {}
    arrays["axis_names"] = np.asarray(result.axis_names, dtype=object)
    for index, values in enumerate(result.axis_values):
        arrays[f"axis_{index}_{_sanitize_key(result.axis_names[index])}"] = np.asarray(values)
    for name, values in result.pathways.items():
        arrays[f"pathway_{_sanitize_key(name)}"] = np.asarray(values)
    for name, values in result.components.items():
        arrays[f"component_{_sanitize_key(name)}"] = np.asarray(values)
    arrays["pathway_names"] = np.asarray(tuple(result.pathways), dtype=object)
    arrays["component_names"] = np.asarray(tuple(result.components), dtype=object)
    arrays["coherence_orders_json"] = np.asarray(
        json.dumps(_jsonable(result.coherence_orders), indent=2),
        dtype=object,
    )
    arrays["fixed_coordinates_json"] = np.asarray(
        json.dumps(_jsonable(result.fixed_coordinates), indent=2),
        dtype=object,
    )
    arrays["pathway_metadata_json"] = np.asarray(
        json.dumps(_jsonable(result.pathway_metadata), indent=2),
        dtype=object,
    )
    np.savez_compressed(path, **arrays)


def _write_parameter_files(txt_path, json_path, payload):
    json_payload = _jsonable(payload)
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    lines = []
    for key, value in json_payload.items():
        lines.append(f"{key}:")
        rendered = json.dumps(value, indent=2)
        lines.append(rendered)
        lines.append("")
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def _write_observables_csv(path, run_id, observables):
    row = {"run_id": run_id}
    for key, value in observables.items():
        if key.endswith("_window") or isinstance(value, (dict, list, tuple)):
            row[key] = json.dumps(_jsonable(value))
        else:
            row[key] = value
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def save_spectrum_bundle(
    result,
    *,
    output_root,
    run_id,
    params=None,
    observables=None,
    note=None,
    figures=None,
    figure_formats=("png",),
):
    """Save raw spectra, figures, parameters, observables, and a mechanism note."""
    output_root = Path(output_root)
    data_dir = output_root / "Data"
    figures_dir = output_root / "Figures"
    analysis_dir = output_root / "Analysis"
    for directory in (data_dir, figures_dir, analysis_dir):
        directory.mkdir(parents=True, exist_ok=True)

    paths = {}
    spectra_path = data_dir / f"{run_id}_spectra.npz"
    _save_raw_spectra(spectra_path, result)
    paths["raw_spectra"] = spectra_path

    if observables is not None:
        observables_path = analysis_dir / f"{run_id}_observables.csv"
        _write_observables_csv(observables_path, run_id, observables)
        paths["observables"] = observables_path

    if note is not None:
        note_path = analysis_dir / f"{run_id}_mechanism_note.md"
        note_path.write_text(str(note), encoding="utf-8")
        paths["mechanism_note"] = note_path

    if figures:
        saved_figures = {}
        for label, figure in figures.items():
            label_key = _sanitize_key(label)
            for extension in figure_formats:
                figure_path = figures_dir / f"{run_id}_{label_key}.{extension}"
                figure.savefig(figure_path, dpi=300, bbox_inches="tight")
                saved_figures[f"{label_key}_{extension}"] = figure_path
        paths["figures"] = saved_figures

    parameter_payload = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "params": {} if params is None else params,
        "observable_windows": {
            key: value for key, value in (observables or {}).items()
            if key.endswith("_window")
        },
        "generated_files": paths,
    }
    parameter_txt = data_dir / f"{run_id}_parameters.txt"
    parameter_json = data_dir / f"{run_id}_parameters.json"
    _write_parameter_files(parameter_txt, parameter_json, parameter_payload)
    paths["parameters_txt"] = parameter_txt
    paths["parameters_json"] = parameter_json
    return paths
