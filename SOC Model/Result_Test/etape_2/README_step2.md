# Etape 2 - Usage

Le code principal est disponible en deux formats:

- `step2_separate_soc_phonon_dimerization.ipynb`
- `step2_separate_soc_phonon_dimerization.py`

Le modele est une extension directe de l'etape 1 avec:

```text
V_BD_eff = V0 + lambda_delta * delta + lambda_C * C1
```

Changer `active_case_key` pour lancer un seul scenario a la fois:

```python
active_case_key = "soc_constant"
active_case_key = "dimerisation_control"
active_case_key = "spin_correlation_control"
```

Les resultats seront ecrits dans:

- `Data/` pour les matrices `S`;
- `Figures/` pour les spectres PDF;
- `Summaries/` pour les parametres et resumes texte/JSON;
- `Analysis/` pour l'analyse de phase 2.

Les dossiers de resultats scientifiques sont volontairement vides dans cette
preparation. Le protocole conceptuel est dans `step2_validation_protocol.tex`.
