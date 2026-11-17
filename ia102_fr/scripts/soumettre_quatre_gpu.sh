#!/bin/bash
#SBATCH --job-name=cnn-fashion-ddp
#SBATCH --account=def-VOTRE_GROUPE
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16                  # 4 cœurs par GPU
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env && source $SLURM_TMPDIR/env/bin/activate
pip install --no-index torch torchvision
cp -r ~/scratch/donnees $SLURM_TMPDIR/

torchrun --standalone --nproc_per_node=4 entrainer_ddp.py \
    --donnees $SLURM_TMPDIR/donnees --epoques 10 --workers 4
