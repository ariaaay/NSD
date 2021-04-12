#!/bin/sh

#SBATCH --job-name=ae
#SBATCH -p tarrq
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=3-00:00:00
#SBATCH --error=/home/yuanw3/error_log/job.%J.err
#SBATCH --output=/home/yuanw3/error_log/job.%J.out

set -eu
source venv/bin/activate

python code/train_autoencoder.py --roi $1 --roi_num $2