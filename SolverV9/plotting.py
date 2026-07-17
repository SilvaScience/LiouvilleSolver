"""Plotting helpers for SolverV8."""

import math
from pathlib import Path
import re

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

from .export import extract_spectrum_profile
from .models import PathwayPlotResult, SpectrumResult


class SpectroscopyPlotter:
    """Plot three pathways and their sum on a shared 2D grid."""

    _COMPONENTS = {
        "real": ("Real", np.real, "bwr", None),
        "imag": ("Imaginary", np.imag, "bwr", None),
        "abs": ("Absolute", np.abs, "magma", 0),
    }
    _COMPONENT_ALIASES = {
        "all": "all", "real": "real", "re": "real",
        "imag": "imag", "imaginary": "imag", "im": "imag",
        "abs": "abs", "absolute": "abs", "magnitude": "abs",
    }

    def __init__(self, w_list=None, detection_phase=np.pi / 2):
        self.w_list = (
            None if w_list is None else np.asarray(w_list, dtype=float)
        )
        self.detection_phase = 0.0 if detection_phase is None else float(detection_phase)

    def _apply_detection_phase(self, data):
        return np.exp(1j * self.detection_phase) * np.asarray(data)

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
        raise ValueError(f"{name}_values must be a one- or two-dimensional array.")

    @staticmethod
    def _index_range(values, lower, upper):
        values = np.asarray(values, dtype=float)
        lo, hi = sorted((float(lower), float(upper)))
        indices = np.flatnonzero((values >= lo) & (values <= hi))
        if indices.size == 0:
            raise ValueError(
                f"No axis samples fall inside bounds ({lower}, {upper})."
            )
        return slice(int(indices[0]), int(indices[-1]) + 1)

    @staticmethod
    def _signed_log_scale(values):
        values = np.asarray(values, dtype=float)
        return np.sign(values) * np.log10(np.abs(values) + 1.0)

    @classmethod
    def _normalize_panel(cls, values):
        scale = float(np.max(np.abs(values)))
        if not np.isfinite(scale):
            raise ValueError("Spectrum contains non-finite values.")
        if scale == 0:
            return values
        return values / scale

    def _prepare_contour_spectra(
        self,
        spectra_list,
        x_values,
        y_values,
        *,
        plot_quadrant,
        zoom_bounds,
        invert_y,
    ):
        if spectra_list is None:
            raise ValueError("spectra_list is required.")
        spectra = [np.asarray(item) for item in spectra_list]
        if not spectra:
            raise ValueError("spectra_list must contain at least one spectrum.")

        x_axis = self._axis_vector(x_values, name="x")
        y_axis = self._axis_vector(y_values, name="y")
        expected_shape = (y_axis.size, x_axis.size)
        for index, values in enumerate(spectra):
            if values.shape != expected_shape:
                raise ValueError(
                    f"Spectrum {index} has shape {values.shape}; "
                    f"expected {expected_shape}."
                )

        quadrant_key = str(plot_quadrant).lower()
        if quadrant_key == "zoom":
            if zoom_bounds is None:
                raise ValueError("zoom_bounds is required for plot_quadrant='Zoom'.")
            if len(zoom_bounds) != 4:
                raise ValueError(
                    "zoom_bounds must contain (x_min, x_max, y_min, y_max)."
                )
            x_slice = self._index_range(x_axis, zoom_bounds[0], zoom_bounds[1])
            y_slice = self._index_range(y_axis, zoom_bounds[2], zoom_bounds[3])
        elif quadrant_key in {"1", "2", "3", "4"}:
            x_zero = int(np.argmin(np.abs(x_axis)))
            y_zero = int(np.argmin(np.abs(y_axis)))
            x_slices = {
                "1": slice(x_zero, None),
                "2": slice(None, x_zero + 1),
                "3": slice(None, x_zero + 1),
                "4": slice(x_zero, None),
            }
            y_slices = {
                "1": slice(y_zero, None),
                "2": slice(y_zero, None),
                "3": slice(None, y_zero + 1),
                "4": slice(None, y_zero + 1),
            }
            x_slice = x_slices[quadrant_key]
            y_slice = y_slices[quadrant_key]
        elif quadrant_key == "all":
            x_slice = slice(None)
            y_slice = slice(None)
        else:
            raise ValueError("plot_quadrant must be 'All', 'Zoom', '1', '2', '3', or '4'.")

        x_axis = x_axis[x_slice]
        y_axis = y_axis[y_slice]
        spectra = [values[y_slice, x_slice] for values in spectra]
        if invert_y:
            y_axis = -y_axis[::-1]
            spectra = [np.flip(values, axis=0) for values in spectra]

        scan_range = [
            float(np.min(x_axis)),
            float(np.max(x_axis)),
            float(np.min(y_axis)),
            float(np.max(y_axis)),
        ]
        return x_axis, y_axis, spectra, scan_range

    @staticmethod
    def _draw_diagonals(ax, scan_range, diagonals, *, color="black"):
        if diagonals[0]:
            ax.plot(
                [scan_range[0], scan_range[1]],
                [scan_range[3], scan_range[2]],
                "--",
                color=color,
                linewidth=0.5,
            )
        if diagonals[1]:
            ax.plot(
                [scan_range[0], scan_range[1]],
                [scan_range[2], scan_range[3]],
                "--",
                color=color,
                linewidth=0.5,
            )

    @staticmethod
    def _plot_subplot(ax, w, data, levels, cmap, title, xlabel, ylabel, vmin):
        limit = np.max(np.abs(data))
        if not np.isfinite(limit):
            raise ValueError("Spectrum contains non-finite values.")
        if limit == 0:
            limit = np.finfo(float).eps
        lower = -limit if vmin is None else vmin
        contour = ax.contourf(w, w, data, levels, cmap=cmap, vmin=lower, vmax=limit)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.figure.colorbar(contour, ax=ax)

    def plot_1d(
        self,
        signal,
        w=None,
        component="real",
        title="1D spectrum",
        xlabel=r"$\omega$",
        ylabel=None,
        label=None,
        reference_positions=None,
        peak_positions=None,
        normalize=False,
        save_path=None,
        show=True,
        ax=None,
    ):
        """Plot one complex one-dimensional spectrum.

        Parameters
        ----------
        signal : array_like
            Complex spectrum sampled on ``w``.
        w : array_like or None
            Frequency axis. Defaults to the axis supplied to the plotter.
        component : {"real", "imag", "abs"}
            Quadrature displayed after applying ``detection_phase``.
        reference_positions : sequence or None
            Expected resonance positions, drawn as dashed vertical lines.
        peak_positions : sequence or None
            Extracted resonance positions, drawn on the spectrum.
        normalize : bool
            Divide the displayed quadrature by its maximum absolute value.
        """
        w = self.w_list if w is None else np.asarray(w, dtype=float)
        signal = np.asarray(signal)
        if w.ndim != 1 or w.size == 0:
            raise ValueError("w must be a non-empty one-dimensional array.")
        if signal.ndim != 1 or signal.shape != w.shape:
            raise ValueError(
                f"signal has shape {signal.shape}; expected {w.shape}."
            )

        component_key = self._COMPONENT_ALIASES.get(str(component).lower())
        if component_key not in self._COMPONENTS:
            raise ValueError("component must be 'real', 'imag', or 'abs'.")
        component_label, transform, _, _ = self._COMPONENTS[component_key]
        data = transform(self._apply_detection_phase(signal))
        if not np.all(np.isfinite(data)):
            raise ValueError("Spectrum contains non-finite values.")
        if normalize:
            scale = np.max(np.abs(data))
            if scale > 0:
                data = data / scale

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.5))
        else:
            fig = ax.figure
        ax.plot(w, data, color="black", label=label)

        if reference_positions is not None:
            for index, position in enumerate(reference_positions):
                ax.axvline(
                    position,
                    color="tab:red",
                    linestyle="--",
                    alpha=0.8,
                    label="Reference positions" if index == 0 else None,
                )
        if peak_positions is not None:
            peak_positions = np.asarray(peak_positions, dtype=float)
            ax.scatter(
                peak_positions,
                np.interp(peak_positions, w, data),
                color="tab:blue",
                zorder=3,
                label="Extracted positions",
            )

        if ylabel is None:
            ylabel = "Normalized signal" if normalize else f"{component_label} signal"
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.2)
        if label is not None or reference_positions is not None or peak_positions is not None:
            ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax

    def plot_contourf_multi_spectra(
        self,
        spectra_list,
        x_values=None,
        y_values=None,
        labels=None,
        title_list=None,
        scale="linear",
        color_map="Spectral_r",
        abs_color_map=None,
        normalization="panel",
        plot_sum=True,
        plot_quadrant="All",
        invert_y=False,
        diagonals=(True, True),
        zoom_bounds=None,
        levels=12,
        contour_lines=True,
        line_color="black",
        linewidth=0.4,
        line_alpha=0.7,
        save_path=None,
        show=True,
    ):
        """Plot complex spectra with the contour style used by QuDPy helpers.

        Each input spectrum is displayed as one row with real, imaginary, and
        absolute-value panels.  The method applies this plotter's detection
        phase before splitting those components.

        Parameters
        ----------
        spectra_list : sequence of array_like
            Complex spectra with shape ``(len(y_values), len(x_values))``.
        x_values, y_values : array_like or None
            Frequency axes.  One-dimensional axes are preferred.  Two-
            dimensional mesh arrays are accepted for compatibility with older
            plotting helpers.  If omitted, ``self.w_list`` is used for both.
        normalization : {"panel", "row", "none"}
            ``"panel"`` normalizes every real/imag/abs panel independently;
            ``"row"`` normalizes the three panels of each spectrum by a common
            row scale; ``"none"`` keeps raw amplitudes with a shared row
            colorbar.
        plot_quadrant : {"All", "Zoom", "1", "2", "3", "4"}
            Select a quadrant using the axis sample nearest zero, or crop to
            ``zoom_bounds=(x_min, x_max, y_min, y_max)``.
        """
        if x_values is None:
            x_values = self.w_list
        if y_values is None:
            y_values = self.w_list
        if x_values is None or y_values is None:
            raise ValueError("x_values and y_values are required when w_list is not set.")

        levels = int(levels)
        if levels < 2:
            raise ValueError("levels must be at least two.")

        x_axis, y_axis, spectra, scan_range = self._prepare_contour_spectra(
            spectra_list,
            x_values,
            y_values,
            plot_quadrant=plot_quadrant,
            zoom_bounds=zoom_bounds,
            invert_y=invert_y,
        )
        spectra = [self._apply_detection_phase(values) for values in spectra]

        if title_list is None:
            row_titles = [str(index + 1) for index in range(len(spectra))]
        else:
            row_titles = [str(title) for title in title_list]
            if len(row_titles) != len(spectra):
                raise ValueError("title_list must match spectra_list length.")

        if plot_sum:
            spectra = [*spectra, sum(spectra)]
            row_titles = [*row_titles, "Total"]

        scale_key = str(scale).lower()
        if scale_key not in {"linear", "log"}:
            raise ValueError("scale must be 'linear' or 'log'.")

        rows = []
        for spectrum in spectra:
            real = np.real(spectrum)
            imag = np.imag(spectrum)
            absolute = np.abs(spectrum)
            if scale_key == "log":
                real = self._signed_log_scale(real)
                imag = self._signed_log_scale(imag)
                absolute = np.log10(absolute + 1.0)
            rows.append([real, imag, absolute])

        normalization_key = str(normalization).lower()
        if normalization_key not in {"panel", "row", "none"}:
            raise ValueError("normalization must be 'panel', 'row', or 'none'.")
        if normalization_key == "panel":
            rows = [
                [self._normalize_panel(values) for values in row]
                for row in rows
            ]
        elif normalization_key == "row":
            normalized_rows = []
            for row in rows:
                row_scale = max(float(np.max(np.abs(values))) for values in row)
                if not np.isfinite(row_scale):
                    raise ValueError("Spectrum contains non-finite values.")
                if row_scale == 0:
                    row_scale = 1.0
                normalized_rows.append([values / row_scale for values in row])
            rows = normalized_rows
        else:
            for row in rows:
                if not all(np.all(np.isfinite(values)) for values in row):
                    raise ValueError("Spectrum contains non-finite values.")

        with plt.rc_context({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 12,
        }):
            figure, axes = plt.subplots(
                len(rows),
                4,
                figsize=(20, 5 * len(rows)),
                gridspec_kw={"width_ratios": [1, 1, 1, 0.06]},
                squeeze=False,
            )

            component_titles = ("real", "imag", "abs")
            x_label, y_label = labels if labels else (r"$\omega_3$", r"$\omega_1$")
            abs_color_map = color_map if abs_color_map is None else abs_color_map

            for row_index, row in enumerate(rows):
                if normalization_key in {"panel", "row"}:
                    signed_norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
                    abs_norm = Normalize(vmin=0.0, vmax=1.0)
                    colorbar_norm = signed_norm
                    ticks = [-1, -0.5, 0, 0.5, 1]
                else:
                    signed_limit = max(
                        float(np.max(np.abs(values))) for values in row[:2]
                    )
                    if signed_limit == 0:
                        signed_limit = np.finfo(float).eps
                    abs_limit = float(np.max(row[2]))
                    if abs_limit == 0:
                        abs_limit = np.finfo(float).eps
                    signed_norm = TwoSlopeNorm(
                        vmin=-signed_limit, vcenter=0.0, vmax=signed_limit
                    )
                    abs_norm = Normalize(vmin=0.0, vmax=abs_limit)
                    colorbar_norm = signed_norm
                    ticks = np.linspace(-signed_limit, signed_limit, 5)

                for column, values in enumerate(row):
                    ax = axes[row_index, column]
                    norm = abs_norm if column == 2 else signed_norm
                    cmap = abs_color_map if column == 2 else color_map
                    contour = ax.contourf(
                        x_axis,
                        y_axis,
                        values,
                        levels=levels,
                        cmap=cmap,
                        norm=norm,
                    )
                    if contour_lines:
                        ax.contour(
                            x_axis,
                            y_axis,
                            values,
                            levels=levels,
                            colors=line_color,
                            linewidths=linewidth,
                            alpha=line_alpha,
                        )
                    self._draw_diagonals(
                        ax,
                        scan_range,
                        diagonals,
                        color=line_color,
                    )
                    ax.set_title(f"{row_titles[row_index]} {component_titles[column]}")
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.set_aspect("equal")
                    ax.tick_params(
                        direction="in",
                        length=6,
                        width=1.4,
                        top=True,
                        right=True,
                    )

                cax = axes[row_index, 3]
                scalar_mappable = cm.ScalarMappable(
                    norm=colorbar_norm,
                    cmap=color_map,
                )
                scalar_mappable.set_array([])
                colorbar = figure.colorbar(scalar_mappable, cax=cax)
                colorbar.set_ticks(ticks)

            figure.tight_layout()
            if save_path:
                figure.savefig(save_path, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
        return figure, axes

    def plot_spectrum_result_contours(
        self,
        result,
        spectra="components",
        names=None,
        totals="auto",
        labels=None,
        title_list=None,
        **kwargs,
    ):
        """Plot a ``SpectrumResult`` using ``plot_contourf_multi_spectra``.

        Parameters
        ----------
        result : SpectrumResult
            Two-dimensional solver output.
        spectra : {"components", "pathways"} or mapping
            Select whether to plot component totals or individual pathways.
        names : sequence or None
            Names to extract from ``result.components`` or ``result.pathways``.
            Defaults to all component names, or all pathway names.
        totals : {"auto", "selected", None, False}
            Only used when ``spectra="pathways"``.  It mirrors
            ``plot_pathways_multiorder`` and can add component or selected
            totals to the plotted rows.
        """
        if not isinstance(result, SpectrumResult):
            raise TypeError("result must be a SpectrumResult.")
        if len(result.axis_names) != 2 or len(result.axis_values) != 2:
            raise ValueError("A two-frequency SpectrumResult is required.")

        selected, x_values, y_values, default_labels = self.select_spectrum_result_data(
            result,
            spectra=spectra,
            names=names,
            totals=totals,
        )
        if title_list is None:
            title_list = tuple(selected)
        if labels is None:
            labels = default_labels
        kwargs.setdefault("plot_sum", False)
        return self.plot_contourf_multi_spectra(
            list(selected.values()),
            x_values=x_values,
            y_values=y_values,
            labels=labels,
            title_list=title_list,
            **kwargs,
        )

    def select_spectrum_result_data(
        self,
        result,
        spectra="components",
        names=None,
        totals="auto",
    ):
        """Select plottable spectra and axes from a two-frequency result."""
        if not isinstance(result, SpectrumResult):
            raise TypeError("result must be a SpectrumResult.")
        if len(result.axis_names) != 2 or len(result.axis_values) != 2:
            raise ValueError("A two-frequency SpectrumResult is required.")

        source_key = str(spectra).lower() if isinstance(spectra, str) else None
        if source_key == "components":
            source = result.components
            selected_names = tuple(source) if names is None else tuple(names)
            selected = {name: np.asarray(source[name]) for name in selected_names}
        elif source_key == "pathways":
            selected_names = self._selected_pathway_names(result, names or "all")
            selected = {
                name: np.asarray(result.pathways[name])
                for name in selected_names
            }
            selected.update(self._selected_totals(result, selected_names, totals))
        elif hasattr(spectra, "items"):
            selected = {str(name): np.asarray(values) for name, values in spectra.items()}
        else:
            raise ValueError("spectra must be 'components', 'pathways', or a mapping.")

        y_values = np.asarray(result.axis_values[0], dtype=float)
        x_values = np.asarray(result.axis_values[1], dtype=float)
        expected_shape = (y_values.size, x_values.size)
        for name, values in selected.items():
            if values.shape != expected_shape:
                raise ValueError(
                    f"Spectrum {name!r} has shape {values.shape}; "
                    f"expected {expected_shape}."
                )

        labels = (
            result.axis_names[1],
            result.axis_names[0],
        )
        return selected, x_values, y_values, labels

    def plot_spectrum_profile(
        self,
        result,
        *,
        spectra="pathways",
        names=("R1", "R2"),
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
        ax=None,
        label=None,
        show=True,
        **plot_kwargs,
    ):
        """Plot a one-dimensional profile extracted from a spectrum result."""
        profile = extract_spectrum_profile(
            result,
            spectra=spectra,
            names=names,
            detection_phase=self.detection_phase,
            quantity=quantity,
            cut=cut,
            window=window,
            center=center,
            half_length=half_length,
            frequency=frequency,
            width=width,
            diagonal=diagonal,
            quadrant=quadrant,
            offset=offset,
            num_points=num_points,
        )
        if ax is None:
            figure, ax = plt.subplots(figsize=plot_kwargs.pop("figsize", (6, 4)))
        else:
            figure = ax.figure

        if label is None:
            label = f"{profile['cut']} {profile['quantity']}"
        ax.plot(profile["axis"], profile["intensity"], label=label, **plot_kwargs)
        ax.set_xlabel(profile["axis_name"])
        ax.set_ylabel(f"{profile['quantity']} intensity")
        if label:
            ax.legend()
        ax.tick_params(direction="in", top=True, right=True)
        figure.tight_layout()
        if show:
            plt.show()
        return figure, ax, profile

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
                f"Unknown pathway name(s) {unknown}; available={available}."
            )
        if len(set(selected)) != len(selected):
            raise ValueError("pathways contains duplicate names.")
        if not selected:
            raise ValueError("At least one pathway must be selected.")
        return selected

    @staticmethod
    def _selected_totals(result, selected_names, totals):
        if totals is None or totals is False:
            return {}
        if totals == "selected":
            return {
                "Selected total": sum(
                    np.asarray(result.pathways[name]) for name in selected_names
                )
            }
        if totals == "auto":
            if set(selected_names) != set(result.pathways):
                return {
                    "Selected total": sum(
                        np.asarray(result.pathways[name])
                        for name in selected_names
                    )
                }
            nq_totals = [
                name for name in result.components
                if re.fullmatch(r"\d+Q", str(name))
            ]
            if nq_totals:
                totals = nq_totals
            else:
                components = []
                for name in selected_names:
                    metadata = result.pathway_metadata.get(name, {})
                    component = metadata.get("component")
                    if component in result.components and component not in components:
                        components.append(component)
                totals = components or list(result.components)
        elif isinstance(totals, str):
            totals = (totals,)

        selected_totals = {}
        for name in totals:
            if name not in result.components:
                raise KeyError(
                    f"Unknown component total {name!r}; "
                    f"available={tuple(result.components)}."
                )
            selected_totals[f"Total {name}"] = np.asarray(
                result.components[name]
            )
        return selected_totals

    @staticmethod
    def _render_ufss_pathway_diagrams(
        result,
        pathway_names,
        *,
        diagram_size,
        display_diagrams,
        save_pdf,
        output_directory,
    ):
        try:
            import ufss
        except ImportError as exc:
            raise ImportError(
                "UFSS is required to render pathway diagrams."
            ) from exc

        generator = ufss.DiagramGenerator(detection_type="polarization")
        generator.diagram_size = diagram_size
        generator.include_state_labels = True
        generator.include_pulse_labels = True
        generator.include_emission_arrow = True

        diagrams = {}
        diagram_paths = {}
        diagram_directory = None
        if save_pdf:
            diagram_directory = Path(output_directory) / "Feynman_diagrams"
            diagram_directory.mkdir(parents=True, exist_ok=True)

        for name in pathway_names:
            metadata = result.pathway_metadata.get(name)
            if metadata is None or not metadata.get("ufss_diagram"):
                raise ValueError(
                    f"Pathway {name!r} has no retained UFSS diagram metadata."
                )
            diagram = tuple(
                (str(interaction), int(pulse_index))
                for interaction, pulse_index in metadata["ufss_diagram"]
            )
            pulse_count = max((item[1] for item in diagram), default=-1) + 1
            generator.pulse_labels = [
                str(index) for index in range(1, pulse_count + 1)
            ]
            generator.draw_diagram(diagram)
            canvas = generator.c
            diagrams[name] = canvas

            if display_diagrams:
                try:
                    from IPython.display import display
                except ImportError as exc:
                    raise ImportError(
                        "IPython is required when display_diagrams=True."
                    ) from exc
                display(canvas, exclude="image/png")

            if save_pdf:
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
                pdf_path = diagram_directory / f"{safe_name}.pdf"
                canvas.writePDFfile(str(pdf_path.with_suffix("")))
                diagram_paths[name] = pdf_path

        return diagrams, diagram_paths

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
        include_diagrams=True,
        diagram_size="medium",
        display_diagrams=False,
        save_pdf=False,
        output_directory=None,
        spectrum_pdf_name="pathway_spectra.pdf",
        show=True,
    ):
        """Plot arbitrary-order pathway spectra and matching UFSS diagrams.

        Individual spectra and component totals come from ``SpectrumResult``.
        Retained UFSS instructions are rendered with the same pathway names.
        PDF output is opt-in; when ``save_pdf`` is false no directory or file
        is created by this method.
        """
        if not isinstance(result, SpectrumResult):
            raise TypeError("result must be a SpectrumResult.")
        if len(result.axis_names) != 2 or len(result.axis_values) != 2:
            raise ValueError("A two-frequency SpectrumResult is required.")
        if save_pdf and output_directory is None:
            raise ValueError(
                "output_directory is required when save_pdf=True."
            )

        selected_names = self._selected_pathway_names(result, pathways)
        selected_totals = self._selected_totals(
            result, selected_names, totals
        )
        panel_data = {
            **{name: np.asarray(result.pathways[name]) for name in selected_names},
            **selected_totals,
        }

        y_values = np.asarray(result.axis_values[0], dtype=float)
        x_values = np.asarray(result.axis_values[1], dtype=float)
        expected_shape = (y_values.size, x_values.size)
        for name, values in panel_data.items():
            if values.shape != expected_shape:
                raise ValueError(
                    f"Panel {name!r} has shape {values.shape}; "
                    f"expected {expected_shape}."
                )

        view_key = self._COMPONENT_ALIASES.get(str(view).lower())
        if view_key not in self._COMPONENTS:
            raise ValueError("view must be 'real', 'imag', or 'abs'.")
        view_label, transform, cmap, absolute_vmin = self._COMPONENTS[view_key]
        phased = {
            name: transform(self._apply_detection_phase(values))
            for name, values in panel_data.items()
        }
        if not all(np.all(np.isfinite(values)) for values in phased.values()):
            raise ValueError("A pathway spectrum contains non-finite values.")

        normalization = str(normalization).lower()
        if normalization not in {"shared", "individual", "none"}:
            raise ValueError(
                "normalization must be 'shared', 'individual', or 'none'."
            )
        global_scale = max(
            (float(np.max(np.abs(values))) for values in phased.values()),
            default=0.0,
        )
        if global_scale == 0:
            global_scale = np.finfo(float).eps
        if normalization == "shared":
            display_data = {
                name: values / global_scale for name, values in phased.items()
            }
            shared_limit = 1.0
            colorbar_label = f"Normalized {view_label.lower()} signal"
        else:
            display_data = dict(phased)
            shared_limit = global_scale if normalization == "none" else None
            colorbar_label = f"{view_label} signal"
            if normalization == "individual":
                for name, values in display_data.items():
                    scale = float(np.max(np.abs(values)))
                    if scale > 0:
                        display_data[name] = values / scale
                colorbar_label = f"Individually normalized {view_label.lower()} signal"

        panel_names = tuple(display_data)
        panel_count = len(panel_names)
        levels = int(levels)
        if levels < 2:
            raise ValueError("levels must be at least two.")
        if ncols is None:
            ncols = int(math.ceil(math.sqrt(panel_count)))
        ncols = int(ncols)
        if ncols < 1:
            raise ValueError("ncols must be positive.")
        nrows = int(math.ceil(panel_count / ncols))
        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.2 * ncols, 4.4 * nrows),
            squeeze=False,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axis_labels = {} if axis_labels is None else dict(axis_labels)
        x_label = axis_labels.get(result.axis_names[1], result.axis_names[1])
        y_label = axis_labels.get(result.axis_names[0], result.axis_names[0])

        for axis, name in zip(axes.flat, panel_names):
            values = display_data[name]
            limit = shared_limit
            if limit is None:
                limit = float(np.max(np.abs(values)))
                if limit == 0:
                    limit = np.finfo(float).eps
            vmin = 0.0 if absolute_vmin == 0 else -limit
            contour_levels = np.linspace(vmin, limit, levels)
            contour = axis.contourf(
                x_values,
                y_values,
                values,
                levels=contour_levels,
                cmap=cmap,
                vmin=vmin,
                vmax=limit,
            )
            metadata = result.pathway_metadata.get(name, {})
            if metadata:
                interactions = " ".join(metadata.get("interactions", ()))
                coherences = metadata.get("coherence_orders", ())
                title = f"{name}: {interactions}\nq={coherences}"
            else:
                title = name
            axis.set(title=title, xlabel=x_label, ylabel=y_label)
            figure.colorbar(contour, ax=axis, label=colorbar_label)
        for axis in axes.flat[panel_count:]:
            axis.set_visible(False)

        spectrum_pdf = None
        if save_pdf:
            output_directory = Path(output_directory)
            output_directory.mkdir(parents=True, exist_ok=True)
            spectrum_pdf = output_directory / spectrum_pdf_name
            if spectrum_pdf.suffix.lower() != ".pdf":
                spectrum_pdf = spectrum_pdf.with_suffix(".pdf")
            figure.savefig(spectrum_pdf, bbox_inches="tight")

        diagrams = {}
        diagram_paths = {}
        if include_diagrams:
            diagrams, diagram_paths = self._render_ufss_pathway_diagrams(
                result,
                selected_names,
                diagram_size=diagram_size,
                display_diagrams=display_diagrams,
                save_pdf=save_pdf,
                output_directory=output_directory,
            )

        if show:
            plt.show()
        return PathwayPlotResult(
            figure=figure,
            axes=axes,
            panel_names=panel_names,
            diagrams=diagrams,
            diagram_paths=diagram_paths,
            spectrum_pdf=spectrum_pdf,
        )

