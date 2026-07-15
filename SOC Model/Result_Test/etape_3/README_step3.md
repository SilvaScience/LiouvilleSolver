# Etape 3 - Dynamic dimerisation/phonon coordinate

Le code principal est disponible en deux formats:

- `step3_dynamic_dimerisation_phonon.py`
- `step3_dynamic_dimerisation_phonon.ipynb`

Le modele teste:

```text
H_mix = V_BD_static L_BD + g_Q Q L_BD
V_BD_static = V0 + lambda_delta * delta + lambda_C * C1
Q = a + a^\dagger
```

Changer `STEP3_ACTIVE_CASE` ou `active_case_key` pour lancer un seul scenario:

```python
active_case_key = "static_spin_correlation"
active_case_key = "static_dimerisation"
active_case_key = "dynamic_dimerisation_phonon"
```

Comparaison attendue:

- `static_spin_correlation` et `static_dimerisation` doivent rester equivalents
  si `V_BD_static` est le meme.
- `dynamic_dimerisation_phonon` peut differer parce que `g_Q Q L_BD` couple les
  manifolds phonon et peut generer une signature spectrale dynamique.

Les resultats seront ecrits dans:

- `Data/` pour les matrices `S`;
- `Figures/` pour les spectres PDF;
- `Summaries/` pour les parametres et resumes texte/JSON;
- `Analysis/` pour l'analyse de phase 2;
- `Feynman_diagrams/` si des diagrammes sont ajoutes plus tard.

Les dossiers de resultats scientifiques sont volontairement vides dans cette
preparation. Le protocole conceptuel est dans `step3_validation_protocol.tex`.
