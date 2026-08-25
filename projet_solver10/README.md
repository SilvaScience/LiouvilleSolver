# SolverV10

`projet_solver10` est un moteur générique de spectroscopie. Il ne contient
aucun modèle physique. Un modèle externe fournit ses secteurs, ses blocs
hamiltoniens, ses transitions, son état initial et son observable.

## Dépendances

- NumPy
- SciPy
- Matplotlib (optionnel, pour `SpectroscopyPlotter`)

## Direction des dépendances

```text
modèle externe  --->  contrats de projet_solver10
script/notebook --->  modèle externe + projet_solver10
projet_solver10 --->  aucun modèle physique
```

Le solver ne doit jamais importer une classe comme `SpinOrbitalModel`.

## Contrat minimal d'un modèle

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
        # direction vaut "plus" ou "minus"
        ...

    def observable_blocks(self, observable_name, source):
        ...

    def initial_condition(self, context=None):
        # PureState(...) ou DensityState(...)
        ...

    def collapse_channels(self, context=None):
        # tuple de CollapseChannel(name, rate, operator_blocks)
        ...

    def equilibrium_state(self, context):
        # État de Gibbs construit et sectorisé par le modèle.
        ...

    def requirements(self):
        # ModelRequirements(...) ou dict
        ...

    def capabilities(self):
        # Ancienne API conservée pour compatibilité.
        ...
```

Chaque bloc peut être un tableau NumPy, une matrice creuse SciPy ou un
`scipy.sparse.linalg.LinearOperator`.

`initial_state()` reste accepté comme repli rétrocompatible pour un état pur.
Le solver ne construit jamais lui-même une base thermique, ne choisit pas les
secteurs thermiquement accessibles et n'invente aucun taux dissipatif.

## États initiaux et température

Un état mixte est fourni sous forme de blocs ket/bra :

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

Les blocs intersectoriels permettent les cohérences bright--dark, spinorielles
ou de moment. Le solver vérifie dimensions, hermiticité, trace et positivité
uniquement pour cet état physique initial.

Le contexte thermique est transmis au modèle :

```python
context = ThermodynamicContext(
    temperature=10.0,
    ensemble="canonical",
    k_ensemble="independent",  # ou "global"
)
solver.feed_model(model, context=context)
```

`equilibrium_state(context)` doit retourner l'état thermique exact dans la base
tronquée construite par le modèle.

## Convention de dynamique

Le cœur numérique utilise le générateur temporel

\[
\mathcal A\rho=-i[H,\rho]
+\sum_j\gamma_j\left(
L_j\rho L_j^\dagger-\frac12\{L_j^\dagger L_j,\rho\}
\right).
\]

Un intervalle temporel applique \(e^{t\mathcal A}\). Un intervalle fréquentiel
utilise uniquement le résolvant direct

\[
\left[(\eta-i\omega)I-\mathcal A\right]^{-1}.
\]

Aucun facteur global \(-i\) n'est inclus dans ce résolvant. Le coefficient
perturbatif du pathway porte donc directement la convention
\(i^n(-1)^{n_B}\), comme dans l'expansion temporelle. Aucune FFT temporelle
n'est utilisée par V10.

Les canaux dissipatifs suivent strictement la convention `(L, gamma)` :

```python
CollapseChannel(
    name="bright_to_dark",
    rate=gamma_bd,
    operator_blocks={
        "bright": {"dark": L_dark_from_bright},
    },
)
```

Le taux ne doit pas être inclus une seconde fois dans l'opérateur.

## Plusieurs observables et fluorescence intégrée

La propagation d'un pathway peut être réutilisée pour plusieurs détections :

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

Les noms des canaux GKSL utilisables pour un comptage sont disponibles avec
`solver.jump_channel_names()`. Le résultat historique reste accessible dans
`result.pathways` et `result.components`. Les sorties supplémentaires sont
organisées par observable :

```python
result.observables["population_a"]["P1"]
result.observable_components["fluorescence"]["rephasing"]
```

Une detection d'action peut ajouter le pulse de projection apres un pathway
a trois interactions. Le coeur du pathway est alors propage une seule fois,
puis le quatrieme pulse est applique uniquement dans la branche de detection :

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

`fourth_interaction` accepte aussi une sequence d'interactions alternatives;
leurs contributions sont alors sommees avec les prefacteurs ket/bra
appropries. La forme historique reste valide : si le quatrieme pulse est deja
present dans `FrequencyPathway.interactions`, il suffit de ne pas fournir
`fourth_interaction` dans l'observable. Les deux formulations donnent le meme
signal.

Une observable `operator` calcule `Tr[O rho]` à la fin du pathway. Une
observable `jump_rate` calcule le taux instantané

\[
I_j(t)=\eta\,\gamma_j\operatorname{Tr}[L_j^\dagger L_j\rho(t)],
\]

et `integrated_jump` en calcule la moyenne intégrée sur la fenêtre demandée.
La fenêtre est mesurée après l'état final du pathway, donc après la dernière
interaction et la dernière propagation du protocole. Il s'agit d'un nombre
moyen de sauts; cette API ne génère pas encore de
trajectoires quantiques ni de distribution de comptage `P(N)`.

L'adaptateur `EigenbasisKModel` accepte aussi plusieurs opérateurs nommés via
`observable_op_arrays={"P_a": ..., "P_b": ..., "P_2X": ...}`. Les opérateurs
doivent être fournis dans la même base que le hamiltonien; l'adaptateur les
transforme ensuite dans la base propre.

La décomposition du dipôle peut rester automatique, ou être fournie
explicitement pour suivre les manifolds UFSS :

```python
model = EigenbasisKModel(
    H_model=H,
    interaction_op_array=mu,
    j_plus_array=J_plus,
    j_minus_array=J_minus,
)
```

Les deux tableaux explicites doivent être fournis ensemble et peuvent être
en `(d, d)` ou `(N_k, d, d)`. Sans ces arguments, la séparation actuelle par
le signe de `Delta_E` et `rwa_tol` reste inchangée.

## Plotter V10

Le plotter est séparé du calcul et accepte une configuration dictionnaire. La
convention est `axis_values[0]` sur l'axe vertical et `axis_values[1]` sur
l'axe horizontal, donc typiquement excitation verticale et émission
horizontale :

```python
import numpy as np
from projet_solver10 import SpectroscopyPlotter

# Pour la convention 2D usuelle : real = absorptive, imag = dispersive.
plotter = SpectroscopyPlotter(detection_phase=np.pi / 2)
plot_params = {
    "source": "pathways",
    "pathways": ["R3", "R1", "R2"],
    "totals": ["1Q"],
    "view": "all",                 # "real", "imag", "abs" ou "all"
    "diagonals": "auto",            # y=-x (rephasing) ou y=x (non-rephasing)
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

Pour tracer plusieurs pathways avec une seule composante :

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

Pour afficher les trois quadratures côte à côte, avec les mêmes pathways :

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

Pour une sortie supplémentaire, utiliser `source="observables"` et fournir
`observable="fluorescence"`. La méthode `plot_pathways_multiorder` reste
disponible comme interface courte inspirée de V9.

## Construction

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

- secteurs opaques fournis par le modèle;
- blocs hamiltoniens couplant éventuellement plusieurs secteurs;
- propagation temporelle par Krylov;
- branches ket/bra de rang un;
- résolvant de Hilbert matrix-free par GMRES;
- pathways fréquentiels exacts par action de Liouville matrix-free;
- aucune matrice de Liouville explicite.

Le chemin fréquentiel utilise encore un vecteur densité de taille \(D^2\).
Il sert de référence exacte et de sanity check, mais ne constitue pas la voie
scalable finale pour les grands espaces spin-orbitaux. Celle-ci demandera un
algorithme bas-rang ou une formulation par fonctions d'onde plus spécialisée.

### `DenseLiouvilleBackend`

- backend de référence pour les petits systèmes;
- espace de Liouville explicite;
- états purs ou mixtes;
- dynamique unitaire ou Lindblad;
- intervalles temporels et fréquentiels;
- utile pour valider ultérieurement le backend sectorisé.

### Backends non couverts par ce release

`LowRankLiouvilleBackend`, ainsi que les moteurs tenpy (`TenpyDMRGEngine`,
`TenpyTDVPEngine`) et l'orchestration many-body associée
(`ManyBodySolver`, `ManyBodyDynamicsSolver`), vivent dans
`projet_solver10/experimental/` et ne sont ni importés ni exposés par
`SpectroscopySolver` dans ce release. Ils restent importables explicitement
(ex. `from projet_solver10.experimental.low_rank_liouville import
LowRankLiouvilleBackend`) pour qui veut continuer à y travailler, mais ne
sont pas maintenus ni garantis stables ici.

## Conventions de pathway

- `Ku`: transition positive sur le ket;
- `Kd`: transition négative sur le ket;
- `Bu`: multiplication à droite par l'opérateur négatif, donc transition
  positive sur le vecteur bra;
- `Bd`: multiplication à droite par l'opérateur positif, donc transition
  négative sur le vecteur bra.

Un `Interaction` peut imposer un secteur source et un secteur cible. En leur
absence, le backend applique tous les blocs déclarés par le modèle.

## Limites actuelles

- la construction de l'état thermique exact reste une responsabilité du modèle;
- seule la forme GKSL/Lindblad est acceptée pour la dynamique non unitaire;
- pas de générateur Redfield général, HEOM ou mémoire de bain;
- le résolvant fréquentiel matrix-free utilise encore un vecteur de taille
  \(D^2\), sans matrice \(D^2\times D^2\), dans `SparseSectorBackend`;
- la déflation explicite du mode stationnaire et la pseudo-inverse de Drazin ne
  sont pas encore disponibles;
- ce release ne couvre que `DenseLiouvilleBackend` et `SparseSectorBackend`;
  voir « Backends non couverts par ce release » ci-dessus;
- la construction et la validation physique de la base appartiennent au modèle.
