# Aide-mémoire IA101 — Apprentissage machine avec Scikit-learn

## Le cycle d'apprentissage

| Étape | Rôle | Exemple |
| --- | --- | --- |
| Modèle | Fonction paramétrée | ŷ = a x + b |
| Perte | Mesure de l'erreur | EQM = (1/n) Σ (ŷᵢ − yᵢ)² |
| Optimisation | Minimiser la perte | Descente de gradient : θ ← θ − η ∇L(θ) |
| Évaluation | Mesurer la généralisation | Erreur sur un ensemble de test jamais vu |

- **Paramètres** : estimés pendant l'entraînement. **Hyperparamètres** : fixés avant (k, `max_depth`, taux d'apprentissage η).
- Gradient de l'EQM pour ŷ = a x + b : ∂L/∂a = (2/n) Σ (ŷᵢ − yᵢ) xᵢ ; ∂L/∂b = (2/n) Σ (ŷᵢ − yᵢ).
- η trop petit : convergence lente. η trop grand : divergence.

## Familles de problèmes

| Famille | Cible | Méthodes |
| --- | --- | --- |
| Classification | Catégorie | `KNeighborsClassifier`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier` |
| Régression | Valeur numérique | `LinearRegression`, `Ridge`, `KNeighborsRegressor`, `RandomForestRegressor` |
| Non supervisé | Aucune | `KMeans`, `PCA` |

## Démarche type

```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

X_ent, X_test, y_ent, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier())
grille = {"kneighborsclassifier__n_neighbors": [1, 3, 5, 9, 15]}      # <étape>__<paramètre>
recherche = GridSearchCV(pipeline, grille, cv=5).fit(X_ent, y_ent)  # sélection sur l'entraînement
recherche.score(X_test, y_test)                                      # évaluation finale, une seule fois
```

## Métriques

| Métrique | Question | Fonction |
| --- | --- | --- |
| Exactitude | Proportion de prédictions correctes | `accuracy_score` |
| Précision | VP / (VP + FP) : les alertes sont-elles fiables ? | `precision_score` |
| Rappel | VP / (VP + FN) : les cas positifs sont-ils détectés ? | `recall_score` |
| Matrice de confusion | Quel type d'erreurs ? | `ConfusionMatrixDisplay` |
| EQM, EAM | Erreur moyenne en régression | `mean_squared_error`, `mean_absolute_error` |
| Silhouette, ARI | Qualité d'un regroupement | `silhouette_score`, `adjusted_rand_score` |

Classes déséquilibrées : ne pas se fier à l'exactitude seule. Le seuil de décision (`predict_proba`) arbitre entre précision et rappel selon le coût des erreurs.

## Diagnostic et actions

| Symptôme | Diagnostic | Action |
| --- | --- | --- |
| Scores d'entraînement et de validation faibles | Sous-apprentissage | Modèle plus complexe, meilleures variables |
| Entraînement nettement supérieur à la validation | Surapprentissage | Modèle plus simple, régularisation, plus de données |
| Variables d'échelles différentes | Distances biaisées | `StandardScaler` dans un `Pipeline` |
| Validation encore croissante sur la courbe d'apprentissage | Données insuffisantes | Collecter davantage de données |

## Non supervisé

- **k-moyennes** : minimise l'inertie Σ ‖xᵢ − μ_c(i)‖² ; normaliser ; choisir k avec le coude, la silhouette et la connaissance du domaine.
- **ACP** : composantes non corrélées, ordonnées par variance expliquée ; `PCA(n_components=0.9)` conserve 90 % de la variance.

## Trois erreurs à éviter

1. Évaluer un modèle sur ses données d'entraînement.
2. Choisir des hyperparamètres en consultant l'ensemble de test.
3. Ajuster un prétraitement sur toutes les données avant le découpage (utiliser un `Pipeline`).

**Suite : IA102 — Apprentissage profond avec PyTorch.**
