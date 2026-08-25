# QuDPy-FDGF

**QuDPy-FDGF: A Python-Based Tool for Computing Ultrafast Nonlinear Optical
Responses Using Frequency-Domain Green's Functions.**

`projet_solver10` remains the compatibility name of the Python package.

`projet_solver10` is a generic spectroscopy engine. It contains no physical
model. An external model supplies its sectors, Hamiltonian blocks,
transitions, initial state, and observable.

## Dependencies

- NumPy
- SciPy
- Matplotlib (optional, for `SpectroscopyPlotter`)

## Dependency direction

```text
external model  --->  projet_solver10 contracts
script/notebook --->  external model + projet_solver10
projet_solver10 --->  no physical model
```

The solver must never import a class such as `SpinOrbitalModel`.

## Minimal model contract

```python
class MyModel:
    def sectors(self):
        ...

    def dimension(self, sector):
        ...

    def hamiltonian_blocks(self, source):
        # {target_sector: operator(target, source)}
        ...

    def transition_blocks(self, operator_name, direction, source):
        # direction is "plus" or "minus"
        ...

    def observable_blocks(self, observable_name, source):
        ...

    def initial_condition(self, context=None):
        # PureState(...) or DensityState(...)
        ...

    def collapse_channels(self, context=None):
        # Tuple of CollapseChannel(name, rate, operator_blocks)
        ...

    def equilibrium_state(self, context):
        # Gibbs state built and partitioned into sectors by the model.
        ...

    def requirements(self):
        # ModelRequirements(...) or dict
        ...

    def capabilities(self):
        # Legacy API retained for compatibility.
        ...
```

Each block may be a NumPy array, a SciPy sparse matrix, or a
`scipy.sparse.linalg.LinearOperator`.

`initial_state()` remains available as a backward-compatible fallback for a
pure state. The solver never builds a thermal basis, chooses thermally
accessible sectors, or invents dissipative rates.

## Initial states and temperature

A mixed state is supplied as ket/bra blocks:

```python
rho0 = DensityState(
    blocks={
        ("bright", "bright"): rho_bb,
        ("dark", "dark"): rho_dd,
        ("bright", "dark"): rho_bd,
        ("dark", "bright"): rho_bd.conj().T,
    }
)
```

Cross-sector blocks support bright--dark, spinor, and momentum coherences. The
solver validates dimensions, Hermiticity, trace, and positivity only for this
physical initial state.

The thermal context is passed to the model:

```python
context = ThermodynamicContext(
    temperature=10.0,
    ensemble="canonical",
    k_ensemble="independent",  # or "global"
)
solver.feed_model(model, context=context)
```

`equilibrium_state(context)` must return the exact thermal state in the
truncated basis built by the model.

## Dynamics convention

The numerical core uses the time-domain generator

\[
\mathcal A\rho=-i[H,\rho]
+\sum_j\gamma_j\left(
L_j\rho L_j^\dagger-\frac12\{L_j^\dagger L_j,\rho\}
\right).
\]

A time interval applies \(e^{t\mathcal A}\). A frequency interval uses only the
direct resolvent

\[
\left[(\eta-i\omega)I-\mathcal A\right]^{-1}.
\]

No global \(-i\) factor is included in this resolvent. The pathway's
perturbative coefficient therefore directly carries the
\(i^n(-1)^{n_B}\) convention, as in the time-domain expansion. No temporal FFT
is used.

Dissipative channels strictly follow the `(L, gamma)` convention:

```python
CollapseChannel(
    name="bright_to_dark",
    rate=gamma_bd,
    operator_blocks={
        "bright": {"dark": L_dark_from_bright},
    },
)
```

The rate must not be included a second time in the operator.

## Multiple observables and integrated fluorescence

A pathway propagation can be reused for multiple detection schemes:

```python
from projet_solver10 import ObservableSpec

fluorescence = ObservableSpec.mean_jump(
    "fluorescence",
    channel="radiative",
    time_window=(0.0, 200.0),
    efficiency=0.35,
    n_steps=401,
)

result = solver.generate_spectrum(
    protocol,
    axes,
    observables={
        "population_a": "P_a",
        "population_b": "P_b",
        "fluorescence": fluorescence,
    },
)
```

The names of GKSL channels available for jump counting are returned by
`solver.jump_channel_names()`. The legacy result remains available through
`result.pathways` and `result.components`. Additional outputs are organized by
observable:

```python
result.observables["population_a"]["P1"]
result.observable_components["fluorescence"]["rephasing"]
```

Action detection can add a projection pulse after a three-interaction pathway.
The pathway core is then propagated only once, and the fourth pulse is applied
only in the detection branch:

```python
action_population = ObservableSpec.action(
    "action_population",
    fourth_interaction={
        "label": "Bu",
        "operator": "light_matter",
        "pulse_index": 3,
    },
    operator="excited_population",
)

action_fluorescence = ObservableSpec.mean_jump(
    "action_fluorescence",
    channel="radiative",
    time_window=(0.0, 200.0),
    fourth_interaction="Bu",
)
```

`fourth_interaction` also accepts a sequence of alternative interactions. Their
contributions are summed with the appropriate ket/bra prefactors. The legacy
form remains valid: if the fourth pulse is already included in
`FrequencyPathway.interactions`, simply omit `fourth_interaction` from the
observable. Both formulations produce the same signal.

An `operator` observable computes `Tr[O rho]` at the end of the pathway. A
`jump_rate` observable computes the instantaneous rate

\[
I_j(t)=\eta\,\gamma_j\operatorname{Tr}[L_j^\dagger L_j\rho(t)],
\]

and `integrated_jump` computes its integrated mean over the requested window.
The window starts after the pathway's final state, and therefore after the
protocol's last interaction and propagation. This is a mean jump count; the
API does not yet generate quantum trajectories or a counting distribution
`P(N)`.

The `EigenbasisKModel` adapter also accepts multiple named operators through
`observable_op_arrays={"P_a": ..., "P_b": ..., "P_2X": ...}`. Operators must
be supplied in the same basis as the Hamiltonian; the adapter then transforms
them to the eigenbasis.

The dipole decomposition may remain automatic or be supplied explicitly to
follow the UFSS manifolds:

```python
model = EigenbasisKModel(
    H_model=H,
    interaction_op_array=mu,
    j_plus_array=J_plus,
    j_minus_array=J_minus,
)
```

Both explicit arrays must be supplied together and may have shape `(d, d)` or
`(N_k, d, d)`. Without these arguments, the current separation based on the
sign of `Delta_E` and `rwa_tol` remains unchanged.

## Plotter V10

The plotter is separate from the computation and accepts a configuration
dictionary. By convention, `axis_values[0]` is plotted on the vertical axis and
`axis_values[1]` on the horizontal axis, typically giving vertical excitation
and horizontal emission:

```python
import numpy as np
from projet_solver10 import SpectroscopyPlotter

# Standard 2D convention: real = absorptive, imag = dispersive.
plotter = SpectroscopyPlotter(detection_phase=np.pi / 2)
plot_params = {
    "source": "pathways",
    "pathways": ["R3", "R1", "R2"],
    "totals": ["1Q"],
    "view": "all",                 # "real", "imag", "abs", or "all"
    "diagonals": "auto",            # y=-x (rephasing) or y=x (non-rephasing)
    "normalization": "row",
    "labels": ("Emission energy (eV)", "Excitation energy (eV)"),
    "title": "2D spectroscopy",
    "show": True,
    "style": {
        "cmap": "RdYlBu_r",
        "abs_cmap": "magma",
        "levels": 30,
        "contour_lines": True,
    },
}
plot_result = plotter.plot_spectrum_result(result, plot_params)
```

To plot multiple pathways with a single component:

```python
plotter.plot_pathways(
    result,
    {
        "pathways": ["R1", "R2", "R3"],
        "detection_phase": np.pi / 2,
        "diagonals": "auto",
        "view": "real",
        "totals": False,
        "normalization": "row",
        "show": True,
    },
)
```

To display all three quadratures side by side for the same pathways:

```python
plotter.plot_real_imag_abs(
    result,
    {
        "pathways": ["R1", "R2", "R3"],
        "detection_phase": np.pi / 2,
        "diagonals": "auto",
        "normalization": "row",
        "style": {"cmap": "RdYlBu_r", "abs_cmap": "magma"},
        "show": True,
    },
)
```

For an additional output, use `source="observables"` and supply
`observable="fluorescence"`. The `plot_pathways_multiorder` method remains
available as a compact V9-inspired interface.

## Setup

```python
from projet_solver10 import SpectroscopySolver
from my_spin_orbital_model import SpinOrbitalModel

model = SpinOrbitalModel(...)

solver = SpectroscopySolver(
    backend="sparse_sector",
    eta=0.02,
    krylov_tolerance=1e-10,
)
solver.feed_model(model)
```

## Backends

### `SparseSectorBackend`

- opaque sectors supplied by the model;
- Hamiltonian blocks that may couple multiple sectors;
- Krylov time propagation;
- rank-one ket/bra branches;
- matrix-free Hilbert-space resolvent solved with GMRES;
- exact frequency-domain pathways through matrix-free Liouville action;
- no explicit Liouville matrix.

The frequency-domain path still uses a density vector of size \(D^2\). It
serves as an exact reference and sanity check, but it is not the final scalable
approach for large spin-orbital spaces. Such systems will require a low-rank
algorithm or a more specialized wave-function formulation.

### `DenseLiouvilleBackend`

- reference backend for small systems;
- explicit Liouville space;
- pure or mixed states;
- unitary or Lindblad dynamics;
- time- and frequency-domain intervals;
- useful for validating the sectorized backend.

### Backends not covered by this release

`LowRankLiouvilleBackend`, the tenpy engines (`TenpyDMRGEngine` and
`TenpyTDVPEngine`), and the associated many-body orchestration
(`ManyBodySolver` and `ManyBodyDynamicsSolver`) live in `experimental/`. They
are neither imported nor exposed by `SpectroscopySolver` in this release. They
remain available through explicit imports (for example,
`from projet_solver10.experimental.low_rank_liouville import
LowRankLiouvilleBackend`) for continued development, but they are not
maintained or guaranteed to be stable here.

## Pathway conventions

- `Ku`: positive transition on the ket;
- `Kd`: negative transition on the ket;
- `Bu`: right multiplication by the negative operator, producing a positive
  transition on the bra vector;
- `Bd`: right multiplication by the positive operator, producing a negative
  transition on the bra vector.

An `Interaction` may specify a source sector and a target sector. If they are
omitted, the backend applies every block declared by the model.

## Current limitations

- construction of the exact thermal state remains the model's responsibility;
- only the GKSL/Lindblad form is supported for non-unitary dynamics;
- no general Redfield generator, HEOM, or bath memory;
- the matrix-free frequency-domain resolvent in `SparseSectorBackend` still
  uses a vector of size \(D^2\), without a \(D^2\times D^2\) matrix;
- explicit stationary-mode deflation and the Drazin pseudoinverse are not yet
  available;
- this release covers only `DenseLiouvilleBackend` and
  `SparseSectorBackend`; see "Backends not covered by this release" above;
- construction and physical validation of the basis belong to the model.
