#!/bin/sh

#SBATCH --job-name=enco
#SBATCH -p tarrq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --mem=30G
#SBATCH --time=10-00:00:00
#SBATCH --error=/home/yuanw3/error_log/job.%J.err
#SBATCH --output=/home/yuanw3/error_log/job.%J.out

set -eu
source venv/bin/activate
module load cuda-10.1
module load cudnn-10.1-v7.6.5.32

task=$1

sub=1

echo "running taskonomy $task task on subject $sub"
python nsd_code/run_modeling.py --model taskrepr_$task --subj $sub --fix_testing --notest
