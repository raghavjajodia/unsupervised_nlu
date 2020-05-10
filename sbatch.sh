#!/bin/bash
#SBATCH --output=/data/rj1408/ptb_wsj_pos/models/base_lstm_defpar/a/train_logs.out
#SBATCH --error=/data/rj1408/ptb_wsj_pos/models/base_lstm_defpar/a/train_logs.err
#SBATCH --exclude=lion3,lion17
#SBATCH --job-name=defpar
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=8GB
#SBATCH --mail-type=END
#SBATCH --mail-user=rj1408@nyu.edu

module purge
module load cuda/9.0.176

eval "$(conda shell.bash hook)"
conda activate dgl_env
srun python3 LM_LatentVariable.py --dataroot /data/rj1408/ptb_wsj_pos/ \
    --batchSize 64 --outf /data/rj1408/ptb_wsj_pos/models/base_lstm_defpar/a/ \
    --cuda
