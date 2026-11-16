#!/bin/bash
#SBATCH --job-name=cnn-fashion
#SBATCH --account=def-VOTRE_GROUPE
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

module load python/3.11                     # vérifier les versions disponibles : module avail python

# Environnement virtuel sur le disque local du nœud, paquets fournis par l'Alliance
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision

# Données copiées sur le disque local du nœud
cp -r ~/scratch/donnees $SLURM_TMPDIR/

torchrun --standalone --nproc_per_node=1 entrainer_ddp.py \
    --donnees $SLURM_TMPDIR/donnees --epoques 10 --workers $SLURM_CPUS_PER_TASK
