#!/bin/sh

#SBATCH --job-name=2b_clip
#SBATCH -p tarrq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --nodelist=mind-1-32
#SBATCH --ntasks=2
#SBATCH --mem=80G
#SBATCH --time=10-00:00:00
#SBATCH --error=/home/yuanw3/error_log/job.%J.err
#SBATCH --output=/home/yuanw3/error_log/job.%J.out

source ~/.bashrc
conda activate conda-env
source ~/NSD/nsdvenv/bin/activate

# model=$1
model="clip laion2b_clip"
sub=5

echo "running $model on subject $sub"
python code/run_modeling.py --model $model --subj $sub --fix_testing
