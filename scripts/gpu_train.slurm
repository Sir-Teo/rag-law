#!/bin/bash
#SBATCH -p a100_short,a100_long,radiology
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=3-00:00:00
#SBATCH --job-name=rag
#SBATCH --output=/gpfs/scratch/wz1492/rag-law/logs/train-%J.log
#SBATCH --exclude=a100-4020

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi
nvcc --version

# activate conda environment
module load gcc/8.1.0
source ~/.bashrc 
conda activate rag

python src/main.py --rerank