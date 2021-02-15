#!/bin/sh

#SBATCH --job-name=tl
#SBATCH -p gpu
#SBATCH --ntasks=2
#SBATCH --mem=100G
#SBATCH --time=10-00:00:00
#SBATCH --error=/home/yuanw3/error_log/job.%J.err
#SBATCH --output=/home/yuanw3/error_log/job.%J.out

set -eu
source venv/bin/activate
module load cuda-11.1.1
module load cudnn-11.1.1-v8.0.4.30

# layer=$1
task=$1
# echo "running taskonomy $task task on subject $subj"

python code/run_modeling.py --model taskrepr_$task --layer input_layer1 --subj 1 --fix_testing --output_dir /user_data/yuanw3/project_outputs/NSD/output --features_dir /user_data/yuanw3/project_outputs/NSD/features --get_features_only
python code/run_modeling.py --model taskrepr_$task --layer input_layer2 --subj 1 --fix_testing --output_dir /user_data/yuanw3/project_outputs/NSD/output --features_dir /user_data/yuanw3/project_outputs/NSD/features --get_features_only
python code/run_modeling.py --model taskrepr_$task --layer input_layer3 --subj 1 --fix_testing --output_dir /user_data/yuanw3/project_outputs/NSD/output --features_dir /user_data/yuanw3/project_outputs/NSD/features --get_features_only
python code/run_modeling.py --model taskrepr_$task --layer input_layer4 --subj 1 --fix_testing --output_dir /user_data/yuanw3/project_outputs/NSD/output --features_dir /user_data/yuanw3/project_outputs/NSD/features --get_features_only
python code/run_modeling.py --model taskrepr_$task --layer input_layer5 --subj 1 --fix_testing --output_dir /user_data/yuanw3/project_outputs/NSD/output --features_dir /user_data/yuanw3/project_outputs/NSD/features --get_features_only
