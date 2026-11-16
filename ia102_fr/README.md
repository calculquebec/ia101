# IA102 — Apprentissage profond avec PyTorch

**Calcul Québec** · Atelier en ligne · 3 heures, de 13 h 30 à 16 h 30 · En français
Formatrice : Gaëlle Talotsing, Calcul Québec

L'atelier présente les fondamentaux de l'apprentissage profond : construire des réseaux de neurones capables d'apprendre automatiquement à partir de données brutes, et les entraîner avec PyTorch.

## Prérequis

Statistiques, programmation Python, analyse de données avec NumPy, pandas et Matplotlib, et bases de l'apprentissage machine (IA101, PYT101, PYT102, DAT201, DAT203 ou équivalent).

## Objectifs pédagogiques

| # | Objectif annoncé | À la fin de l'atelier, les participants seront en mesure de… | Notebooks |
| --- | --- | --- | --- |
| O1 | Comprendre les bases de l'apprentissage profond | expliquer le neurone artificiel, les activations, la rétropropagation et la raison d'être des réseaux profonds | 0, 1 |
| O2 | Créer des réseaux de neurones | définir un réseau avec `nn.Sequential` ou `nn.Module`, choisir la sortie et la perte, construire un réseau convolutif | 1, 3 |
| O3 | Entraîner des modèles avec PyTorch | écrire la boucle d'entraînement, utiliser `Dataset` et `DataLoader`, améliorer la généralisation (régularisation, transfert d'apprentissage) | 2, 3, 4 |
| O4 | S'initier à l'entraînement haute performance | utiliser un GPU, optimiser le chargement des données, employer la précision mixte et les points de contrôle, lancer un entraînement multi-GPU et le soumettre avec Slurm | 3, 5 |

## Contenu et déroulement

| Heure | Partie du programme | Notebook | Contenu | Exercices |
| --- | --- | --- | --- | --- |
| 13 h 30 | Révision de IA101 | `00_revision_ia101_et_motivation` | Cycle d'apprentissage, règle de mise à jour, généralisation ; limites des modèles linéaires | Test de positionnement |
| 13 h 45 | Introduction aux réseaux de neurones | `01_introduction_aux_reseaux_de_neurones` | Neurone, activations, couches, tenseurs, rétropropagation calculée à la main et vérifiée, `nn.Module` | 1, 2, 3 |
| 14 h 15 | Entraînement avec PyTorch | `02_entrainer_un_reseau_avec_pytorch` | Boucle d'entraînement, taux d'apprentissage, SGD et Adam, `zero_grad()` | 4, 5 |
| 14 h 45 | Pause | | | |
| 14 h 55 | Entraînement avec PyTorch | `03_reseaux_convolutifs` | Convolution, pooling, `Dataset` et `DataLoader`, CNN sur GPU, analyse des erreurs | 6, 7, 8 |
| 15 h 30 | Entraînement avec PyTorch | `04_ameliorer_la_generalisation` | Surapprentissage, arrêt précoce, dropout, augmentation de données, transfert d'apprentissage | 9, 10 |
| 15 h 50 | Introduction à l'entraînement haute performance | `05_entrainement_haute_performance` | Débit, chargement parallèle, précision mixte, points de contrôle, DDP, scripts Slurm ; test final | 11, 12, 13 |
| 16 h 20 | Synthèse | | Questions, évaluation | |
| Autonome | Prolongement de O1 | `bonus_transformers` | Plongements, encodage positionnel, attention, Transformers | B1 |

## Contenu du dossier

```
IA102/
├── 00_… à 05_…, bonus_transformers.ipynb   notebooks des participants (exercices à compléter, solutions repliables)
├── solutions/                              corrigés complets et exécutés (version formatrice)
├── scripts/                                entrainer_ddp.py, soumettre_un_gpu.sh, soumettre_quatre_gpu.sh
├── preparer_donnees.py                     téléchargement préalable des données et des poids
├── aide-memoire_IA102.md                   fiche de synthèse
├── requirements.txt
└── README.md
```

## Préparation de l'environnement

Les **nœuds de calcul de l'Alliance n'ont pas accès à Internet**. Depuis un nœud de connexion :

```bash
python preparer_donnees.py --donnees ~/scratch/donnees     # FashionMNIST, CIFAR-10, poids de ResNet18 (environ 250 Mo)
export IA102_DONNEES=~/scratch/donnees
```

Session JupyterHub recommandée : 1 GPU, 4 cœurs, 16 Go de mémoire. Tout fonctionne aussi sur processeur, plus lentement. Deux écrans sont recommandés (Zoom et JupyterLab).

## Temps d'exécution et résultats de référence (processeur, 2 cœurs, sans GPU)

| Notebook | Durée | Résultats des corrigés |
| --- | --- | --- |
| 0 à 2 | moins de 20 s chacun | Cercles : 50 % (régression logistique) contre 100 % (réseau) ; vins : 100 % |
| 3 | environ 3 min | CNN : 89,1 % sur le test après 3 époques |
| 4 | environ 3 min, plus 2 min pour le transfert | 500 images : écart de perte validation − entraînement ramené de 0,80 à 0,24 par la régularisation |
| 5 | environ 1 min | Chargement : 1,6 s → 0,9 s avec 2 processus ; DDP : 188 lots en 7,0 s (1 processus) contre 94 lots en 4,0 s par processus (2 processus) |

Sur GPU, chaque entraînement prend quelques secondes.

## Notes pour l'animation

- Alterner démonstration (10 minutes au plus), pratique guidée et exercice individuel ; utiliser les sondages Zoom pour les cellules « Prédiction ».
- Notebook 2 : l'exercice 4 (ordonner la boucle) est central ; faire énoncer les cinq étapes par les participants.
- Notebook 3 : commenter la matrice de confusion pendant l'entraînement.
- Notebook 4, section 5 : nécessite les poids de ResNet18 et CIFAR-10. **Vérifier avant l'atelier** que `preparer_donnees.py` a été exécuté ; sinon la section affiche un message et se poursuit sans erreur. Les résultats de cette section n'ont pas pu être mesurés lors de la préparation (pas d'accès aux poids ImageNet dans l'environnement de construction) : exécuter la section une fois sur la grappe pour connaître les valeurs à annoncer.
- Notebook 5 : la précision mixte et le DDP ne montrent leur gain réel que sur GPU ; si possible, exécuter ces cellules en direct sur un nœud GPU. Remplacer `def-VOTRE_GROUPE` par le compte réel et vérifier la version du module Python (`module avail python`).
