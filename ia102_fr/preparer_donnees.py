"""Télécharge à l'avance les données et les poids nécessaires à l'atelier IA102.

À exécuter une fois sur une machine connectée à Internet : un poste personnel, ou un
nœud de connexion d'une grappe de l'Alliance (les nœuds de calcul n'ont pas accès à Internet).

    python preparer_donnees.py                               # données dans ./donnees
    python preparer_donnees.py --donnees ~/scratch/donnees

Indiquez ensuite le dossier aux notebooks :
    export IA102_DONNEES=~/scratch/donnees

Les poids de ResNet18 sont placés dans le cache de PyTorch ($TORCH_HOME, par défaut ~/.cache/torch).
Utilisez la même valeur de TORCH_HOME lors du téléchargement et pendant l'atelier.
"""
import argparse
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights

parser = argparse.ArgumentParser()
parser.add_argument("--donnees", default="donnees")
args = parser.parse_args()

print("1/3 FashionMNIST (notebooks 3, 4 et 5)")
datasets.FashionMNIST(args.donnees, train=True, download=True)
datasets.FashionMNIST(args.donnees, train=False, download=True)

print("2/3 CIFAR-10 (notebook 4, transfert d'apprentissage)")
datasets.CIFAR10(args.donnees, train=True, download=True)
datasets.CIFAR10(args.donnees, train=False, download=True)

print("3/3 Poids de ResNet18 préentraîné sur ImageNet (notebook 4)")
resnet18(weights=ResNet18_Weights.DEFAULT)

print(f"Terminé. Données disponibles dans : {args.donnees}")
