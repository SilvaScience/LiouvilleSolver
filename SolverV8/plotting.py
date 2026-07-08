"""Plotting helpers for SolverV8."""

import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

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

    def plot_pathways_grid(
        self,
        pathways_dict,
        signal_type="rephasing",
        total_signal=None,
        w=None,
        levels=20,
        save_path=None,
        show=True,
        zoom_quadrant=True,
        zoom_bounds=None,
        component="all",
    ):
        """Plot R1-R3 or R4-R6 and their sum for selected components."""
        w = self.w_list if w is None else np.asarray(w, dtype=float)
        if w.ndim != 1 or w.size == 0:
            raise ValueError("w must be a non-empty one-dimensional array.")

        signal_key = str(signal_type).lower().replace("-", "")
        signal_definitions = {
            "rephasing": (("R1", "R2", "R3"), -1, "Rephasing"),
            "unrephasing": (("R4", "R5", "R6"), 1, "Non-rephasing"),
            "nonrephasing": (("R4", "R5", "R6"), 1, "Non-rephasing"),
        }
        if signal_key not in signal_definitions:
            raise ValueError("signal_type must be 'rephasing' or 'unrephasing'.")
        pathway_names, y_sign, display_name = signal_definitions[signal_key]

        pathway_data = []
        for name in pathway_names:
            value = pathways_dict.get(name, pathways_dict.get(int(name[1:])))
            if value is None:
                raise KeyError(f"Missing pathway {name!r}.")
            value = np.asarray(value)
            if value.shape != (w.size, w.size):
                raise ValueError(
                    f"Pathway {name!r} has shape {value.shape}; expected {(w.size, w.size)}."
                )
            pathway_data.append(value)

        if total_signal is None:
            total_signal = sum(pathway_data)
        total_signal = np.asarray(total_signal)
        if total_signal.shape != (w.size, w.size):
            raise ValueError(
                f"total_signal has shape {total_signal.shape}; expected {(w.size, w.size)}."
            )

        component_key = self._COMPONENT_ALIASES.get(str(component).lower())
        if component_key is None:
            raise ValueError("component must be 'all', 'real', 'imag', or 'abs'.")
        component_keys = tuple(self._COMPONENTS) if component_key == "all" else (component_key,)

        if zoom_bounds is not None:
            if len(zoom_bounds) != 4:
                raise ValueError("zoom_bounds must contain (x_min, x_max, y_min, y_max).")
            x_min, x_max, y_min, y_max = map(float, zoom_bounds)
            bounds = (x_min, x_max, y_min, y_max)
            if not np.all(np.isfinite(bounds)) or x_min >= x_max or y_min >= y_max:
                raise ValueError("zoom_bounds must be finite and strictly increasing.")

        phased_data = [self._apply_detection_phase(data) for data in (*pathway_data, total_signal)]
        column_titles = (*pathway_names, f"Total {display_name}")
        fig, axes = plt.subplots(
            len(component_keys), 4, figsize=(20, 4.7 * len(component_keys)),
            sharex=True, sharey=True, squeeze=False,
        )

        for row, key in enumerate(component_keys):
            label, transform, cmap, vmin = self._COMPONENTS[key]
            for column, data in enumerate(phased_data):
                ax = axes[row, column]
                self._plot_subplot(
                    ax, w, transform(data), levels, cmap,
                    f"{label} / {column_titles[column]}",
                    r"$\omega_3$" if row == len(component_keys) - 1 else "",
                    r"$\omega_1$" if column == 0 else "", vmin,
                )
                diagonal_extent = min(float(np.max(w)), abs(float(np.min(w))))
                ax.plot(
                    [0, diagonal_extent], [0, y_sign * diagonal_extent],
                    color="white", linestyle="--", linewidth=1.0, alpha=0.7,
                )
                if zoom_bounds is not None:
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                elif zoom_quadrant:
                    ax.set_xlim(0, np.max(w))
                    ax.set_ylim(np.min(w), 0) if y_sign < 0 else ax.set_ylim(0, np.max(w))

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig, axes
