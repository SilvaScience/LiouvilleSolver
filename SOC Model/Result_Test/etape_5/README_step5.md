# Etape 5 - Temperature and field scans

This step turns the Step 4 discrete trend controls into scans.

Scan families:

- `temperature_lattice_dynamic`: sweep `T` at `B=0` with the lattice channel active below `T_SP`.
- `field_lattice_dynamic`: sweep `B` at `T=7` to suppress `T_SP(B)` and the lattice channel.
- `temperature_C1_static`: sweep `T` at `B=0` with a static `C1` channel and `delta=0` contribution.

Run one case at a time with:

```powershell
$env:STEP5_ACTIVE_CASE = "T_lattice_T_7"
python -B "SOC Model\Result_Test\etape_5\step5_temperature_field_scans.py"
```

The analysis script aggregates all generated cases under `Analysis/`.
