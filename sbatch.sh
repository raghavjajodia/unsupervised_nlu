#!/bin/bash
#SBATCH --output=/misc/vlgscratch4/BrunaGroup/rj1408/nlu/ptb_wsj_pos/models/test/a/train_logs.out
#SBATCH --error=/misc/vlgscratch4/BrunaGroup/rj1408/nlu/ptb_wsj_pos/models/test/a/train_logs.err
#SBATCH --exclude=lion3,lion17
#SBATCH --job-name=test
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
srun python3 LM_LatentVariable.py --dataroot /misc/vlgscratch4/BrunaGroup/rj1408/nlu/ptb_wsj_pos/ \
    --batchSize 64 --outf /misc/vlgscratch4/BrunaGroup/rj1408/nlu/ptb_wsj_pos/models/test/a/ \
    --cuda
