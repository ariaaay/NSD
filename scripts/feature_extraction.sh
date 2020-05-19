#!/bin/sh

#SBATCH --job-name=fe
#SBATCH -p tarrq
#SBATCH --ntasks=2
#SBATCH --mem=50G
#SBATCH --time=10-00:00:00
#SBATCH --error=/home/yuanw3/error_log/job.%J.err
#SBATCH --output=/home/yuanw3/error_log/job.%J.out

set -eu
source venv/bin/activate
module load cuda-10.1
module load cudnn-10.1-v7.6.5.32

python code/extract_convnet_features.py $1 --subsample avgpool --model vgg19 --subsampling_size 10000