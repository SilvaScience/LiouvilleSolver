# Etape 4 - Independent temperature and field trends

This step extends the Step 3 dynamic dimerisation/phonon model with independent
temperature and field trends.

The intended logic is:

- `delta(T, B)` turns off above the spin-Peierls transition or when the field
  suppresses the transition temperature.
- `C1(T, B)` remains finite in both the dimerised and uniform regimes.
- A spectral feature that follows `delta` is compatible with a spin-Peierls or
  lattice-controlled channel.
- A feature that persists when `delta = 0` but `C1` is still finite should not
  be attributed uniquely to the dimerised phase.

Run one case at a time with:

```powershell
$env:STEP4_ACTIVE_CASE = "lowT_lattice_dynamic"
python "SOC Model\Result_Test\etape_4\step4_temperature_field_trends.py"
```

Available cases:

- `lowT_lattice_dynamic`
- `highT_lattice_off_C1_finite`
- `highB_lattice_suppressed`
- `highT_C1_static`

Each run saves the PDF spectrum, the complex response arrays, and text/JSON
parameter summaries.
