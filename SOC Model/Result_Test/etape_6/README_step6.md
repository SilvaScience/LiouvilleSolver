# Etape 6 - Dynamic phonon parameter scans

This step isolates the dynamic phonon coordinate by scanning `g_Q` and
`omega_Q` at fixed static bright-dark mixing.

Run one case at a time:

```powershell
$env:STEP6_ACTIVE_CASE = "gQ_0p01"
python -B "SOC Model\Result_Test\etape_6\step6_dynamic_phonon_parameter_scans.py"
```

Recommended cases:

- `gQ_0`, `gQ_0p0025`, `gQ_0p005`, `gQ_0p01`, `gQ_0p015`, `gQ_0p02`
- `omegaQ_0p02`, `omegaQ_0p035`, `omegaQ_0p05`, `omegaQ_0p07`

After all cases are produced, run:

```powershell
python -B "SOC Model\Result_Test\etape_6\Analysis\analyze_step6_results.py"
```
