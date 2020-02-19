#!/bin/sh

#SBATCH --job-name=taskf
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --mem=10G
#SBATCH --time=10-00:00:00
#SBATCH --error=/home/yuanw3/error_log/job.%J.err
#SBATCH --output=/home/yuanw3/error_log/job.%J.out

set -eu
cd /home/yuanw3/taskonomy/taskbank
source taskvenv/bin/activate
module load cuda-10.0
module load cudnn-10.1-v7.6.5.32

STIMULI_DIR="/lab_data/tarrlab/common/datasets/NSD_images"
OUT_DIR="/lab_data/tarrlab/yuanw3/taskonomy_features/genStimuli/"

#n=0
task=$1

#for imgfile in $(ls -1 $STIMULI_DIR/* | sort -r); do
for imgfile in $STIMULI_DIR/*; do
#	n=$((n + 1))
	store_name=$(basename $imgfile)
	target_DIR=${OUT_DIR}${task}

	if ! [ -e $target_DIR ]; then
		mkdir $target_DIR
	fi

  echo "processing $imgfile for task $task"
  tmp="$(cut -d'.' -f1 <<<"$imgfile")"
  id="$(cut -d'/' -f7 <<<"$tmp")"
  printf -v old_name "COCO_train2014_%012d.jpg" $id

  if [ ! -e $target_DIR/$store_name ] && [ ! -e $target_DIR/$old_name ]; then
		python /home/yuanw3/taskonomy/taskbank/tools/run_img_task.py --task $task --img $imgfile --store "$target_DIR/$store_name" --store-rep
	fi

done

#echo $n
