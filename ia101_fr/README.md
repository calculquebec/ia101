# IA101 — Apprentissage machine avec Scikit-learn

**Calcul Québec** · Atelier en ligne (Zoom) · 3 heures, de 13 h 30 à 16 h 30 · En français
Formatrice : Gaëlle Talotsing, Calcul Québec

Créer des modèles prédictifs avec une bibliothèque d'IA accessible. L'atelier explore les principales méthodes de classification, de prédiction et de détection de tendances, et montre comment construire des modèles avec scikit-learn.

## Prérequis

- Bonnes connaissances en statistiques (fonctions et dérivées).
- Programmation Python (PYT101 ou équivalent).
- Analyse et visualisation de données avec Python (DAT201, DAT203 ou équivalent) ; aisance avec NumPy, pandas et Matplotlib.

## Objectifs d'apprentissage

| # | À la fin de l'atelier, les participants seront en mesure de… | Partie du programme | Notebooks |
| --- | --- | --- | --- |
| O1 | situer un problème dans la bonne famille de méthodes (classification, régression, regroupement) | Motivation | 0 |
| O2 | expliquer le fonctionnement de l'apprentissage : modèle, perte, gradient, descente de gradient, généralisation | Introduction aux mathématiques de l'apprentissage machine | 1 |
| O3 | entraîner un modèle de classification ou de régression avec l'interface `fit` / `predict` / `score` | Exercices et applications pratiques | 2, 6 |
| O4 | évaluer un modèle avec une métrique adaptée au problème et au coût des erreurs | Exercices et applications pratiques | 3 |
| O5 | améliorer un modèle de façon rigoureuse : prétraitement, validation croisée, recherche d'hyperparamètres | Exercices et applications pratiques | 4 |
| O6 | détecter des structures dans des données non étiquetées : k-moyennes, analyse en composantes principales | Exercices et applications pratiques | 5 |

## Déroulement

| Heure | Notebook | Contenu | Exercices |
| --- | --- | --- | --- |
| 13 h 30 | `00_motivation` | Programmation classique et apprentissage machine ; familles de problèmes ; test de positionnement | 1 |
| 13 h 40 | `01_mathematiques_de_l_apprentissage` | Modèle, EQM, surface de perte, gradient (dérivation et vérification), descente de gradient, taux d'apprentissage, généralisation | 2, 3, 4 |
| 14 h 20 | `02_premier_modele_de_classification` | Interface scikit-learn, KNN, frontière de décision, normalisation, `Pipeline` | 5, 6 |
| 14 h 50 | Pause | | |
| 15 h 00 | `03_evaluer_un_modele` | Régression logistique, matrice de confusion, précision, rappel, seuil de décision | 7, 8 |
| 15 h 20 | `04_ameliorer_un_modele` | Courbe de validation, validation croisée, `GridSearchCV`, courbe d'apprentissage | 9, 10 |
| 15 h 50 | `05_detecter_des_structures` | k-moyennes, méthode du coude, silhouette, ACP | 11, 12 |
| 16 h 05 | `06_application_integratrice` | Démarche complète sur un problème de régression ; test de positionnement final | 5 étapes |
| 16 h 25 | Synthèse | Questions, évaluation de l'atelier | |

## Contenu du dossier

```
IA101/
├── 00_motivation.ipynb … 06_application_integratrice.ipynb   notebooks des participants (exercices à compléter, solutions repliables)
├── solutions/                                                 corrigés complets et exécutés (version formatrice)
├── aide-memoire_IA101.md                                      fiche de synthèse d'une page
├── requirements.txt
└── README.md
```

Tous les jeux de données sont intégrés à scikit-learn (iris, vins, cancer du sein, chiffres manuscrits, diabète). Aucun téléchargement n'est nécessaire, ce qui permet de travailler sur des nœuds de calcul sans accès à Internet.

## Environnement

- **JupyterHub de l'Alliance :** un cœur et 4 Go de mémoire suffisent ; chaque notebook s'exécute en moins d'une minute.
- **Poste personnel :** `python -m venv ia101 && source ia101/bin/activate && pip install -r requirements.txt && jupyter lab`.
- Deux écrans sont recommandés (Zoom et JupyterLab).

## Conventions des notebooks

| Mention | Signification |
| --- | --- |
| Prédiction | Formuler une hypothèse avant d'exécuter la cellule. |
| Exercice | Compléter les lignes marquées `# À COMPLÉTER` ; une solution repliable suit. |
| Question de vérification | Question à choix multiples avec rétroaction immédiate. |
| À retenir | Synthèse du notebook. |

## Notes pour l'animation

- Alterner démonstration (10 minutes au plus), pratique guidée et exercice individuel.
- Les cellules « Prédiction » se prêtent à un sondage Zoom ou à une réponse dans le clavardage avant exécution.
- Notebook 1 : l'activité d'ajustement manuel par curseurs (3 minutes) précède la descente de gradient ; demander les meilleures EQM dans le clavardage.
- Notebook 6 : utiliser les salles de discussion Zoom (équipes de deux) ; le corrigé obtient une EAM de 41,5 avec une régression Ridge normalisée.
- Valeurs de référence des corrigés : vins 66,7 % puis 93,3 % après normalisation ; cancer du sein 98,6 % (2 faux négatifs au seuil 0,5, 1 au seuil 0,2) ; chiffres manuscrits 96,4 % sur le test ; ARI des k-moyennes sur les iris 0,62.
