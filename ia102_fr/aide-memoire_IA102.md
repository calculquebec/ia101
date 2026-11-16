# Aide-mémoire IA102 — Apprentissage profond avec PyTorch

## Bases

- **Neurone** : a = φ(wᵀx + b). **Couche** : Z = X Wᵀ + b. **Réseau** : enchaînement de couches et d'activations.
- **Activations** : ReLU (couches cachées), sigmoïde (sortie binaire), softmax (sortie multiclasse). Sans non-linéarité, le réseau reste linéaire.
- **Rétropropagation** : règle des fonctions composées, de la sortie vers l'entrée ; `perte.backward()` remplit `.grad`.
- **Mise à jour** : θ ← θ − η ∇L(θ) ; `optimiseur.step()`.

## Définir un réseau

```python
class MonReseau(nn.Module):
    def __init__(self):
        super().__init__()
        self.extracteur = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifieur = nn.Sequential(nn.Flatten(), nn.Linear(16 * 14 * 14, 10))
    def forward(self, x):
        return self.classifieur(self.extracteur(x))
```

| Tâche | Sortie | Perte |
| --- | --- | --- |
| Régression | `Linear(…, 1)` | `nn.MSELoss()` |
| Binaire | `Linear(…, 1)` | `nn.BCEWithLogitsLoss()` |
| Multiclasse | `Linear(…, k)` | `nn.CrossEntropyLoss()` (étiquettes `torch.long`) |

## Boucle d'entraînement

```python
appareil = "cuda" if torch.cuda.is_available() else "cpu"
modele = MonReseau().to(appareil)
optimiseur = torch.optim.Adam(modele.parameters(), lr=1e-3)
chargeur = DataLoader(donnees, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

for epoque in range(n_epoques):
    modele.train()
    for x, y in chargeur:
        x, y = x.to(appareil, non_blocking=True), y.to(appareil, non_blocking=True)
        sortie = modele(x)                # 1. passe avant
        perte = perte_fn(sortie, y)       # 2. perte
        optimiseur.zero_grad()            # 3. réinitialiser les gradients
        perte.backward()                  # 4. rétropropagation
        optimiseur.step()                 # 5. mise à jour
    modele.eval()
    with torch.no_grad():
        ...                               # validation
```

## Réseaux convolutifs

`Conv2d(c_in, c_out, 3, padding=1)` → `ReLU` → `MaxPool2d(2)` (taille divisée par 2), répété ; puis `Flatten` → `Linear`.
Forme d'un lot d'images : (lot, canaux, hauteur, largeur).

## Généralisation

| Symptôme | Action | Mise en œuvre |
| --- | --- | --- |
| Perte de validation qui remonte | Arrêt précoce | Conserver le meilleur modèle, patience |
| Écart entraînement / validation | Dropout | `nn.Dropout(0.5)` |
| Peu d'images | Augmentation | `transforms.RandomHorizontalFlip()`, `RandomAffine` |
| Très peu d'images | Transfert d'apprentissage | `resnet18(weights=ResNet18_Weights.DEFAULT)`, tronc gelé, nouvelle tête |
| Perte immobile | Taux d'apprentissage | Adam, η entre 1e-4 et 1e-2 |

## Entraînement haute performance

| Levier | Mise en œuvre |
| --- | --- |
| Mesurer | Débit en exemples/s ; `torch.cuda.synchronize()` avant de lire l'horloge ; `nvidia-smi` |
| Chargement | `num_workers = $SLURM_CPUS_PER_TASK`, `pin_memory=True`, données sur `$SLURM_TMPDIR` |
| Précision mixte | `with torch.autocast("cuda", dtype=torch.bfloat16):` ; avec `float16`, ajouter `torch.amp.GradScaler("cuda")` |
| Points de contrôle | `torch.save({"epoque": e, "modele": m.state_dict(), "optimiseur": opt.state_dict()}, chemin)` |
| Multi-GPU (DDP) | `torchrun --nproc_per_node=N` ; `dist.init_process_group("nccl")` ; `DDP(modele, device_ids=[rang_local])` ; `DistributedSampler` + `set_epoch` ; sauvegarde par le rang 0 |

## Slurm (Alliance)

```bash
#SBATCH --account=def-groupe  --gpus-per-node=1  --cpus-per-task=4  --mem=16G  --time=00:30:00
module load python
virtualenv --no-download $SLURM_TMPDIR/env && source $SLURM_TMPDIR/env/bin/activate
pip install --no-index torch torchvision
cp -r ~/scratch/donnees $SLURM_TMPDIR/
torchrun --standalone --nproc_per_node=1 entrainer_ddp.py --donnees $SLURM_TMPDIR/donnees --workers $SLURM_CPUS_PER_TASK
```

`sbatch` soumettre · `squeue -u $USER` suivre · `scancel` annuler · `seff` bilan · `salloc` session interactive.
Les nœuds de calcul n'ont pas accès à Internet : télécharger données et poids à l'avance.

## Erreurs fréquentes

- `Expected all tensors to be on the same device` : `.to(appareil)` sur le modèle et sur chaque lot.
- `mat1 and mat2 shapes cannot be multiplied` : taille d'entrée incorrecte de la première `Linear` après `Flatten`.
- `CUDA out of memory` : réduire la taille du lot ou utiliser la précision mixte.
- Perte erratique : vérifier la présence de `zero_grad()`.
