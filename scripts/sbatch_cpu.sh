#!/bin/sh

#SBATCH --job-name=ana
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yuanw3@andrew.cmu.edu
#SBATCH -p tarrq
#SBATCH --ntasks=2
#SBATCH --mem=70G
#SBATCH --time=10-00:00:00
#SBATCH --error=/home/yuanw3/error_log/job.%J.err
#SBATCH --output=/home/yuanw3/error_log/job.%J.out
hostname

singularity shell -B /lab_data,/user_data --nv ~/container_images/pytorch.sif
source ~/.bashrc
conda activate conda-env

python code/analyze_clip_results.py --performance_analysis_by_roi --rerun_df
echo "dataframe done..."
python code/analyze_clip_results.py --group_analysis_by_roi
echo "pca..."
python code/analyze_clip_results.py --group_weight_analysis

