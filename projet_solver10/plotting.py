"""Configurable plotting tools for QuDPy-FDGF results.

All numerical work remains in ``SpectroscopySolver``. This module selects data
from ``SpectrumResult``, applies display conventions, and renders one- or
two-dimensional responses. Matplotlib remains optional so the numerical core
can run without plotting dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

try:  # Keep the solver usable without Matplotlib.
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
except ImportError:  # pragma: no cover - depends on the user environment.
    cm = None
    plt = None
    Normalize = None
    TwoSlopeNorm = None

from .results import PlotResult, SpectrumResult


class SpectroscopyPlotter:
    """Spectrum result and configuration -> rendered plots.

    The first result axis is vertical and the second is horizontal.
    """

    _COMPONENT_ALIASES = {
        "all": "all",
        "real": "real",
        "re": "real",
        "imag": "imag",
        "im": "imag",
        "imaginary": "imag",
        "abs": "abs",
        "absolute": "abs",
        "magnitude": "abs",
    }
    _COMPONENT_LABELS = {"real": "Real", "imag": "Imaginary", "abs": "Absolute"}

    def __init__(self, w_list=None, detection_phase=0.0):
        self.w_list = (
            None if w_list is None else np.asarray(w_list, dtype=float)
        )
        self.detection_phase = (
            0.0 if detection_phase is None else float(detection_phase)
        )

    @staticmethod
    def _require_matplotlib():
        if plt is None:
            raise ImportError(
                "SpectroscopyPlotter requires Matplotlib. "
                "Install matplotlib to use plotting functions."
            )

    @staticmethod
    def _axis_vector(values, *, name):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            return values
        if values.ndim == 2:
            if name == "x":
                return values[0, :]
            if name == "y":
                return values[:, 0]
        raise ValueError(f"{name}_values must be a 1D or 2D array.")

    @staticmethod
    def _index_range(values, lower, upper):
        values = np.asarray(values, dtype=float)
        lo, hi = sorted((float(lower), float(upper)))
        indices = np.flatnonzero((values >= lo) & (values <= hi))
        if indices.size == 0:
            raise ValueError(
                f"No axis sample falls within ({lower}, {upper})."
            )
        return slice(int(indices[0]), int(indices[-1]) + 1)

    @classmethod
    def _normalize_view(cls, view):
        if isinstance(view, str):
            raw = [view]
        else:
            raw = list(view)
        if not raw:
            raise ValueError("view must contain at least one component.")
        values = []
        for item in raw:
            key = cls._COMPONENT_ALIASES.get(str(item).lower())
            if key == "all":
                values.extend(("real", "imag", "abs"))
            elif key in {"real", "imag", "abs"}:
                values.append(key)
            else:
                raise ValueError("view must use real, imag, abs, or all.")
        result = []
        for item in values:
            if item not in result:
                result.append(item)
        return tuple(result)

    @staticmethod
    def _merge_config(params=None, overrides=None):
        if params is None:
            config = {}
        elif isinstance(params, Mapping):
            config = dict(params)
        else:
            raise TypeError("Plotter configuration must be a mapping.")
        if overrides:
            config.update(overrides)
        config["style"] = dict(config.get("style", {}))
        return config

    def _apply_detection_phase(self, values, phase=None):
        phase = self.detection_phase if phase is None else float(phase)
        return np.exp(1j * phase) * np.asarray(values)

    @staticmethod
    def _resolve_diagonals(diagonals, x_values, y_values):
        """Diagonal setting and axes -> anti/main diagonal flags.

        ``auto`` follows the physical signs of standard 1Q axes.
        """
        if diagonals is None:
            return False, False
        if isinstance(diagonals, str):
            key = diagonals.lower().replace("_", "-")
            if key in {"none", "off", "false"}:
                return False, False
            if key in {"both", "all", "true"}:
                return True, True
            if key in {"rephasing", "anti", "anti-diagonal", "negative"}:
                return True, False
            if key in {"nonrephasing", "unrephasing", "main", "diagonal", "positive"}:
                return False, True
            if key in {"auto", "physical"}:
                x_sign = np.sign(np.nanmean(np.asarray(x_values, dtype=float)))
                y_sign = np.sign(np.nanmean(np.asarray(y_values, dtype=float)))
                if x_sign == 0 or y_sign == 0:
                    return True, True
                return (False, True) if x_sign == y_sign else (True, False)
            raise ValueError(
                "diagonals must be False, True, 'auto', 'rephasing', "
                "'nonrephasing', or 'both'."
            )
        if isinstance(diagonals, bool):
            return diagonals, diagonals
        try:
            resolved = tuple(bool(value) for value in diagonals)
        except TypeError as exc:
            raise ValueError("diagonals must contain two booleans.") from exc
        if len(resolved) != 2:
            raise ValueError("diagonals must contain two booleans.")
        return resolved

    @staticmethod
    def _selected_pathway_names(result, pathways):
        available = tuple(result.pathways)
        if pathways is None or pathways == "all":
            return available
        if isinstance(pathways, str):
            pathways = (pathways,)
        selected = tuple(str(name) for name in pathways)
        unknown = [name for name in selected if name not in result.pathways]
        if unknown:
            raise KeyError(
                f"Unknown pathways {unknown}; available: {available}."
            )
        if len(set(selected)) != len(selected):
            raise ValueError("pathways must not contain duplicates.")
        if not selected:
            raise ValueError("At least one pathway must be selected.")
        return selected

    @staticmethod
    def _selected_totals(result, selected_names, totals, *, components=None):
        if totals is None or totals is False:
            return {}
        components = result.components if components is None else components
        if totals == "selected":
            return {
                "Selected total": sum(
                    np.asarray(components[name])
                    for name in selected_names
                    if name in components
                )
            }
        if totals == "auto":
            if set(selected_names) != set(result.pathways):
                return {
                    "Selected total": sum(
                        np.asarray(result.pathways[name]) for name in selected_names
                    )
                }
            totals = [name for name in components if str(name).endswith("Q")]
            if not totals:
                totals = list(components)
        elif isinstance(totals, str):
            totals = (totals,)

        selected_totals = {}
        for name in totals:
            if name not in components:
                raise KeyError(
                    f"Unknown component {name!r}; available: {tuple(components)}."
                )
            selected_totals[f"Total {name}"] = np.asarray(components[name])
        return selected_totals

    def select_spectrum_result_data(
        self,
        result,
        *,
        source="pathways",
        names=None,
        pathways=None,
        totals="auto",
        observable=None,
    ):
        """Spectrum result and selectors -> panels, axes, and labels."""
        if not isinstance(result, SpectrumResult):
            raise TypeError("result must be a SpectrumResult.")
        if len(result.axis_names) != 2 or len(result.axis_values) != 2:
            raise ValueError("A heatmap requires a result with two axes.")

        source_key = str(source).lower()
        if source_key in {"component", "components"}:
            source_data = result.components
            selected_names = tuple(source_data) if names is None else tuple(names)
            selected = {
                str(name): np.asarray(source_data[name]) for name in selected_names
            }
        elif source_key in {"pathway", "pathways"}:
            selected_names = self._selected_pathway_names(
                result, pathways if pathways is not None else names
            )
            selected = {
                name: np.asarray(result.pathways[name]) for name in selected_names
            }
            selected.update(
                self._selected_totals(result, selected_names, totals)
            )
        elif source_key in {"observable", "observables"}:
            if observable is None:
                raise ValueError(
                    "observable is required with source='observables'."
                )
            if observable not in result.observables:
                raise KeyError(
                    f"Unknown observable {observable!r}; "
                    f"available: {tuple(result.observables)}."
                )
            source_data = result.observables[observable]
            selected_names = self._selected_pathway_names(
                result, pathways if pathways is not None else names
            )
            selected = {
                name: np.asarray(source_data[name]) for name in selected_names
            }
            selected.update(
                self._selected_totals(
                    result,
                    selected_names,
                    totals,
                    components=result.observable_components.get(observable, {}),
                )
            )
        elif source_key in {"observable_components", "observable_components"}:
            if observable is None:
                raise ValueError(
                    "observable is required with source='observable_components'."
                )
            try:
                source_data = result.observable_components[observable]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown observable {observable!r}; "
                    f"available: {tuple(result.observable_components)}."
                ) from exc
            selected_names = tuple(source_data) if names is None else tuple(names)
            selected = {
                str(name): np.asarray(source_data[name]) for name in selected_names
            }
        elif isinstance(source, Mapping):
            selected = {
                str(name): np.asarray(values) for name, values in source.items()
            }
        else:
            raise ValueError(
                "source must be 'pathways', 'components', 'observables', "
                "'observable_components', or a mapping."
            )

        y_values = np.asarray(result.axis_values[0], dtype=float)
        x_values = np.asarray(result.axis_values[1], dtype=float)
        expected_shape = (y_values.size, x_values.size)
        for name, values in selected.items():
            if values.shape != expected_shape:
                raise ValueError(
                    f"Panel {name!r} has shape {values.shape}; "
                    f"expected {expected_shape}."
                )
        labels = (result.axis_names[1], result.axis_names[0])
        return selected, x_values, y_values, labels

    @staticmethod
    def _crop_axes(
        panel_data,
        x_values,
        y_values,
        *,
        plot_quadrant="all",
        zoom_bounds=None,
        invert_y=False,
    ):
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        key = str(plot_quadrant).lower()
        if key == "zoom":
            if zoom_bounds is None or len(zoom_bounds) != 4:
                raise ValueError(
                    "zoom_bounds=(x_min, x_max, y_min, y_max) is required."
                )
            x_slice = SpectroscopyPlotter._index_range(
                x_values, zoom_bounds[0], zoom_bounds[1]
            )
            y_slice = SpectroscopyPlotter._index_range(
                y_values, zoom_bounds[2], zoom_bounds[3]
            )
        elif key in {"1", "2", "3", "4"}:
            x_zero = int(np.argmin(np.abs(x_values)))
            y_zero = int(np.argmin(np.abs(y_values)))
            x_slice = {
                "1": slice(x_zero, None),
                "2": slice(None, x_zero + 1),
                "3": slice(None, x_zero + 1),
                "4": slice(x_zero, None),
            }[key]
            y_slice = {
                "1": slice(y_zero, None),
                "2": slice(y_zero, None),
                "3": slice(None, y_zero + 1),
                "4": slice(None, y_zero + 1),
            }[key]
        elif key == "all":
            x_slice = slice(None)
            y_slice = slice(None)
        else:
            raise ValueError("plot_quadrant must be All, Zoom, 1, 2, 3, or 4.")

        x_values = x_values[x_slice]
        y_values = y_values[y_slice]
        panel_data = {
            name: np.asarray(values)[y_slice, x_slice]
            for name, values in panel_data.items()
        }
        if invert_y:
            y_values = -y_values[::-1]
            panel_data = {
                name: np.flip(values, axis=0)
                for name, values in panel_data.items()
            }
        return panel_data, x_values, y_values

    @staticmethod
    def _metadata_title(name, result, show_metadata=True):
        if not show_metadata:
            return str(name)
        metadata = result.pathway_metadata.get(name, {})
        if not metadata:
            return str(name)
        interactions = metadata.get("interactions", ())
        labels = []
        for interaction in interactions:
            if isinstance(interaction, Mapping):
                labels.append(str(interaction.get("label", interaction)))
            else:
                labels.append(str(interaction))
        coherence = metadata.get("coherence_orders", ())
        if labels:
            return f"{name}: {' '.join(labels)}\nq={coherence}"
        return str(name)

    @staticmethod
    def _draw_diagonals(ax, x_values, y_values, diagonals, color="black"):
        if isinstance(diagonals, bool):
            diagonals = (diagonals, diagonals)
        if diagonals[0]:
            ax.plot(
                [x_values[0], x_values[-1]],
                [y_values[-1], y_values[0]],
                "--",
                color=color,
                linewidth=0.6,
            )
        if diagonals[1]:
            ax.plot(
                [x_values[0], x_values[-1]],
                [y_values[0], y_values[-1]],
                "--",
                color=color,
                linewidth=0.6,
            )

    @staticmethod
    def _safe_limit(values):
        limit = float(np.nanmax(np.abs(values)))
        if not np.isfinite(limit):
            raise ValueError("Spectrum contains non-finite values.")
        return limit if limit > 0 else np.finfo(float).eps

    def _render_panel_data(
        self,
        panel_data,
        x_values,
        y_values,
        *,
        config,
        result=None,
        labels=None,
    ):
        self._require_matplotlib()
        if not panel_data:
            raise ValueError("No panel to plot.")
        config = self._merge_config(config)
        style = config["style"]
        views = self._normalize_view(config.get("view", "all"))
        normalization = str(config.get("normalization", "row")).lower()
        normalization = {"individual": "panel", "shared": "global"}.get(
            normalization, normalization
        )
        if normalization not in {"panel", "row", "global", "none"}:
            raise ValueError(
                "normalization must be panel, row, global, or none."
            )
        levels = int(config.get("levels", style.get("levels", 30)))
        if levels < 2:
            raise ValueError("levels must be at least 2.")

        cropped, x_values, y_values = self._crop_axes(
            panel_data,
            x_values,
            y_values,
            plot_quadrant=config.get("plot_quadrant", "all"),
            zoom_bounds=config.get("zoom_bounds"),
            invert_y=bool(config.get("invert_y", False)),
        )
        detection_phase = config.get("detection_phase", self.detection_phase)
        phased = {
            name: self._apply_detection_phase(values, detection_phase)
            for name, values in cropped.items()
        }
        for values in phased.values():
            if not np.all(np.isfinite(values)):
                raise ValueError("A panel contains non-finite values.")

        transformed = {
            name: {
                "real": np.real(values),
                "imag": np.imag(values),
                "abs": np.abs(values),
            }
            for name, values in phased.items()
        }
        if normalization == "global":
            global_limit = max(
                self._safe_limit(values)
                for row in transformed.values()
                for values in row.values()
            )
        else:
            global_limit = None

        cmap = style.get("cmap", "RdYlBu_r")
        abs_cmap = style.get("abs_cmap", "magma")
        contour_lines = bool(style.get("contour_lines", True))
        line_color = style.get("line_color", "black")
        line_width = float(style.get("linewidth", style.get("line_width", 0.35)))
        line_alpha = float(style.get("line_alpha", 0.55))
        colorbar = bool(style.get("colorbar", True))
        aspect = config.get("aspect", "equal")
        title_map = dict(config.get("panel_titles", {}))
        panel_names = tuple(transformed)
        nrows = len(panel_names)
        ncols = len(views)
        figsize = config.get("figsize", (4.8 * ncols, 4.2 * nrows))
        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
            sharex=bool(config.get("sharex", False)),
            sharey=bool(config.get("sharey", False)),
            constrained_layout=bool(config.get("constrained_layout", True)),
        )
        labels = labels or ("x", "y")
        x_label, y_label = labels
        diagonals = self._resolve_diagonals(
            config.get("diagonals", "auto"), x_values, y_values
        )

        for row_index, name in enumerate(panel_names):
            row_values = transformed[name]
            if normalization == "row":
                row_limit = max(self._safe_limit(values) for values in row_values.values())
            else:
                row_limit = None
            for column_index, component in enumerate(views):
                values = row_values[component]
                if normalization == "panel":
                    limit = self._safe_limit(values)
                elif normalization in {"row", "global"}:
                    limit = row_limit if normalization == "row" else global_limit
                else:
                    limit = self._safe_limit(values)

                if component == "abs":
                    norm = Normalize(vmin=0.0, vmax=limit)
                    selected_cmap = abs_cmap
                else:
                    if normalization == "none":
                        signed_limit = self._safe_limit(values)
                    else:
                        signed_limit = limit
                    norm = TwoSlopeNorm(
                        vmin=-signed_limit,
                        vcenter=0.0,
                        vmax=signed_limit,
                    )
                    selected_cmap = cmap

                axis = axes[row_index, column_index]
                contour = axis.contourf(
                    x_values,
                    y_values,
                    values,
                    levels=levels,
                    cmap=selected_cmap,
                    norm=norm,
                )
                if contour_lines:
                    axis.contour(
                        x_values,
                        y_values,
                        values,
                        levels=levels,
                        colors=line_color,
                        linewidths=line_width,
                        alpha=line_alpha,
                    )
                self._draw_diagonals(
                    axis,
                    x_values,
                    y_values,
                    diagonals,
                    color=line_color,
                )
                title = title_map.get(
                    name,
                    self._metadata_title(name, result, config.get("show_metadata", True))
                    if result is not None
                    else str(name),
                )
                if ncols > 1:
                    title = f"{title}\n{self._COMPONENT_LABELS[component]}"
                axis.set_title(title)
                axis.set_xlabel(x_label)
                axis.set_ylabel(y_label)
                axis.set_aspect(aspect)
                axis.tick_params(
                    direction="in",
                    top=True,
                    right=True,
                    labelsize=style.get("tick_labelsize", None),
                )
                if colorbar:
                    label = config.get(
                        "colorbar_label",
                        f"{self._COMPONENT_LABELS[component]} signal",
                    )
                    figure.colorbar(contour, ax=axis, label=label)

        if config.get("title"):
            figure.suptitle(config["title"])
        save_path = config.get("save_path")
        if save_path:
            figure.savefig(save_path, dpi=int(config.get("dpi", 300)), bbox_inches="tight")
        if config.get("show", True):
            plt.show()
        return PlotResult(
            figure=figure,
            axes=axes,
            panel_names=panel_names,
            data=transformed,
            configuration=config,
        )

    def plot_spectrum_result(self, result, params=None, **overrides):
        """Spectrum result and configuration -> rendered plot artifacts."""
        config = self._merge_config(params, overrides)
        selected, x_values, y_values, labels = self.select_spectrum_result_data(
            result,
            source=config.get("source", "pathways"),
            names=config.get("names"),
            pathways=config.get("pathways"),
            totals=config.get("totals", "auto"),
            observable=config.get("observable"),
        )
        label_override = config.get("labels")
        labels = labels if label_override is None else tuple(label_override)
        return self._render_panel_data(
            selected,
            x_values,
            y_values,
            config=config,
            result=result,
            labels=labels,
        )

    def plot_pathways(self, result, params=None, **overrides):
        """Selected pathways and configuration -> panel plot.

        Each pathway defaults to one row showing its real part.
        """
        config = self._merge_config(params, overrides)
        config.setdefault("source", "pathways")
        config.setdefault("pathways", "all")
        config.setdefault("totals", False)
        config.setdefault("view", "real")
        return self.plot_spectrum_result(result, config)

    def plot_real_imag_abs(self, result, params=None, **overrides):
        """Selected pathways -> real, imaginary, and absolute panels.

        Detection phase is applied once before splitting components.
        """
        config = self._merge_config(params, overrides)
        config.setdefault("source", "pathways")
        config.setdefault("pathways", "all")
        config.setdefault("totals", False)
        config["view"] = ("real", "imag", "abs")
        return self.plot_spectrum_result(result, config)

    def plot_spectrum_result_contours(self, result, params=None, **overrides):
        """Explicit alias matching the V9 plotter API."""
        return self.plot_spectrum_result(result, params, **overrides)

    def plot_pathways_multiorder(
        self,
        result,
        pathways="all",
        totals="auto",
        view="real",
        normalization="shared",
        ncols=None,
        levels=30,
        axis_labels=None,
        save_path=None,
        show=True,
        **kwargs,
    ):
        """V9-compatible wrapper for selected pathway panels."""
        config = {
            "source": "pathways",
            "pathways": pathways,
            "totals": totals,
            "view": view,
            "normalization": normalization,
            "levels": levels,
            "labels": axis_labels,
            "save_path": save_path,
            "show": show,
        }
        config.update(kwargs)
        if ncols is not None:
            config.setdefault("ncols", ncols)
        return self.plot_spectrum_result(result, config)

    def plot_contourf_multi_spectra(
        self,
        spectra_list,
        x_values=None,
        y_values=None,
        labels=None,
        title_list=None,
        params=None,
        **overrides,
    ):
        """Complex 2D matrix list -> rendered panels."""
        config = self._merge_config(params, overrides)
        x_values = self.w_list if x_values is None else x_values
        y_values = self.w_list if y_values is None else y_values
        if x_values is None or y_values is None:
            raise ValueError("x_values and y_values are required.")
        x_values = self._axis_vector(x_values, name="x")
        y_values = self._axis_vector(y_values, name="y")
        spectra = [np.asarray(values) for values in spectra_list]
        if not spectra:
            raise ValueError("spectra_list cannot be empty.")
        expected = (y_values.size, x_values.size)
        if any(values.shape != expected for values in spectra):
            raise ValueError(f"Every spectrum must have shape {expected}.")
        names = (
            [str(index + 1) for index in range(len(spectra))]
            if title_list is None
            else [str(value) for value in title_list]
        )
        if len(names) != len(spectra):
            raise ValueError("title_list must match the length of spectra_list.")
        if config.pop("plot_sum", False):
            spectra.append(sum(spectra))
            names.append("Total")
        panel_data = dict(zip(names, spectra))
        config.setdefault("labels", labels or ("x", "y"))
        return self._render_panel_data(
            panel_data,
            x_values,
            y_values,
            config=config,
            labels=config["labels"],
        )

    def plot_1d(
        self,
        signal,
        w=None,
        params=None,
        **overrides,
    ):
        """Complex 1D response -> phase-consistent line plot."""
        self._require_matplotlib()
        config = self._merge_config(params, overrides)
        w = self.w_list if w is None else np.asarray(w, dtype=float)
        signal = np.asarray(signal)
        if w is None or w.ndim != 1 or signal.shape != w.shape:
            raise ValueError("signal and w must be vectors of equal size.")
        view = self._normalize_view(config.get("view", "real"))
        phased = self._apply_detection_phase(
            signal, config.get("detection_phase", self.detection_phase)
        )
        data = {
            component: {
                "real": np.real,
                "imag": np.imag,
                "abs": np.abs,
            }[component](phased)
            for component in view
        }
        if config.get("normalize", False):
            scale = max(self._safe_limit(values) for values in data.values())
            data = {name: values / scale for name, values in data.items()}
        figure, axis = plt.subplots(figsize=config.get("figsize", (8, 4.5)))
        colors = config.get(
            "colors", {"real": "tab:blue", "imag": "tab:orange", "abs": "black"}
        )
        linestyles = config.get(
            "linestyles", {"real": "-", "imag": "-", "abs": "--"}
        )
        for component, values in data.items():
            axis.plot(
                w,
                values,
                color=colors.get(component),
                linestyle=linestyles.get(component, "-"),
                label=config.get("component_labels", {}).get(
                    component, self._COMPONENT_LABELS[component]
                ),
            )
        for position in config.get("reference_positions", ()):
            axis.axvline(position, color="tab:red", linestyle="--", alpha=0.8)
        component = view[0] if len(view) == 1 else "signal"
        axis.set(
            title=config.get("title", "1D spectrum"),
            xlabel=config.get("xlabel", r"$\omega$"),
            ylabel=config.get(
                "ylabel",
                "Normalized signal" if config.get("normalize", False)
                else self._COMPONENT_LABELS.get(component, "Signal"),
            ),
        )
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        if config.get("save_path"):
            figure.savefig(config["save_path"], dpi=int(config.get("dpi", 300)), bbox_inches="tight")
        if config.get("show", True):
            plt.show()
        return PlotResult(
            figure=figure,
            axes=np.asarray([[axis]], dtype=object),
            panel_names=tuple(view),
            data=data,
            configuration=config,
        )
