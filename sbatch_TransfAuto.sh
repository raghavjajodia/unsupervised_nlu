#!/bin/bash
#SBATCH --output=/misc/vlgscratch4/BrunaGroup/rj1408/nlu/ptb_wsj_pos/models/roberta_auto/finetune/a/train_logs.out
#SBATCH --error=/misc/vlgscratch4/BrunaGroup/rj1408/nlu/ptb_wsj_pos/models/roberta_auto/finetune/a/train_logs.err
#SBATCH --exclude=lion3,lion17
#SBATCH --job-name=robertafine
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --mail-type=END
#SBATCH --mail-user=rj1408@nyu.edu

module purge
module load cuda/9.0.176

eval "$(conda shell.bash hook)"
conda activate dgl_env
srun python3 Roberta_LatentVariable.py --dataroot /misc/vlgscratch4/BrunaGroup/rj1408/nlu/ptb_wsj_pos/ \
    --batchSize 1 --ngpu 2 --outf /misc/vlgscratch4/BrunaGroup/rj1408/nlu/ptb_wsj_pos/models/roberta_auto/finetune/a/ \
    --cuda --checkpointf /misc/vlgscratch4/BrunaGroup/rj1408/nlu/ptb_wsj_pos/models/roberta_auto/a \
    --lr 0.00001 --workers 4
