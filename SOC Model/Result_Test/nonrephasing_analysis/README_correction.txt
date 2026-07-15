Corrected nonrephasing plotting note
====================================

Use these files for the corrected nonrephasing interpretation:

- priority_nonrephasing_analysis.pdf
- priority_nonrephasing_zoom_metrics.csv
- priority_nonrephasing_zoom_summary.txt
- spectra_etape_*_rephasing_unrephasing_abs.png

The earlier broad-grid files priority_nonrephasing_summary.txt,
priority_nonrephasing_validation.txt, and priority_nonrephasing_comparison_metrics.csv
were produced before the nonrephasing plotting convention was corrected.  They
sampled the nonrephasing response on omega1 < 0, so they should not be used for
nonrephasing amplitude conclusions.

Corrected convention:

- rephasing: omega1 < 0, omega3 > 0;
- unrephasing: omega1 > 0, omega3 > 0.

The zoom spectra were generated under ZoomData/ with a narrow window centered on
the bright feature.  They are for plotting and finite-window comparison only;
the original scan results were not overwritten.
