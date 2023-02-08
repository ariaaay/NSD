#!/bin/sh

#SBATCH --job-name=fe
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --mem=50G
#SBATCH --time=10-00:00:00
#SBATCH --error=/home/yuanw3/error_log/job.%J.err
#SBATCH --output=/home/yuanw3/error_log/job.%J.out

python code/extract_features_across_models.py --model simclr
python code/extract_features_across_models.py --model clip
python code/extract_features_across_models.py --model blip

