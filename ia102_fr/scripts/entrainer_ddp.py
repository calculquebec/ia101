"""Entraînement multi-GPU avec DistributedDataParallel. Lancement : torchrun --nproc_per_node=<nb GPU> entrainer_ddp.py"""
import os, time, argparse
import torch, torch.nn as nn, torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument("--donnees", default="donnees")
parser.add_argument("--epoques", type=int, default=1)
parser.add_argument("--lot", type=int, default=64, help="taille du lot PAR processus")
parser.add_argument("--limite", type=int, default=0, help="nombre d'images utilisées (0 = toutes)")
parser.add_argument("--workers", type=int, default=0)
args = parser.parse_args()

# 1-2. Un processus par GPU ; torchrun fournit RANK, LOCAL_RANK et WORLD_SIZE
dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
rang, nb_processus = dist.get_rank(), dist.get_world_size()
rang_local = int(os.environ.get("LOCAL_RANK", 0))
if torch.cuda.is_available():
    torch.cuda.set_device(rang_local)
appareil = torch.device(f"cuda:{rang_local}" if torch.cuda.is_available() else "cpu")

# 4. Chaque processus reçoit une partie distincte des données
donnees = datasets.FashionMNIST(args.donnees, train=True, transform=transforms.ToTensor())
if args.limite:
    donnees = Subset(donnees, range(args.limite))
echantillonneur = DistributedSampler(donnees, shuffle=True)
chargeur = DataLoader(donnees, batch_size=args.lot, sampler=echantillonneur, num_workers=args.workers, pin_memory=torch.cuda.is_available())

# 3. Copie du modèle sur chaque GPU, gradients moyennés automatiquement
torch.manual_seed(0)
modele = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10)).to(appareil)
modele = DDP(modele, device_ids=[rang_local] if torch.cuda.is_available() else None)
optimiseur = torch.optim.Adam(modele.parameters(), lr=1e-3)
perte_fn = nn.CrossEntropyLoss()

for epoque in range(args.epoques):
    echantillonneur.set_epoch(epoque)           # mélange différent à chaque époque
    debut = time.time()
    for x, y in chargeur:
        x, y = x.to(appareil, non_blocking=True), y.to(appareil, non_blocking=True)
        optimiseur.zero_grad(); perte = perte_fn(modele(x), y); perte.backward(); optimiseur.step()
    print(f"[rang {rang}/{nb_processus}] époque {epoque + 1} : {len(chargeur)} lots en {time.time() - debut:.1f} s", flush=True)

# 5. Sauvegarde par le rang 0 uniquement
if rang == 0:
    torch.save(modele.module.state_dict(), "cnn_ddp.pt")
    print("Modèle sauvegardé par le rang 0.")
dist.destroy_process_group()
