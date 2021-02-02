#!/bin/sh

#SBATCH --job-name=taskf
#SBATCH -p tarrq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --mem=10G
#SBATCH --time=10-00:00:00
#SBATCH --error=/home/yuanw3/error_log/job.%J.err
#SBATCH --output=/home/yuanw3/error_log/job.%J.out

set -eu
cd /home/yuanw3/taskonomy/taskbank
source taskvenv/bin/activate
module load cuda-10.1
module load cudnn-10.1-v7.6.5.32

STIMULI_DIR="/lab_data/tarrlab/common/datasets/NSD_images"
OUT_DIR="/lab_data/tarrlab/yuanw3/taskonomy_features/genStimuli_layers/"

#n=0
task=$1
target_DIR=${OUT_DIR}${task}
if ! [ -e $target_DIR ]; then
	mkdir $target_DIR
fi

i=0
while read p; do
	echo "$p $i/10000"
	file_name=$p
	store_name="$p.jpg"
	img_file="$STIMULI_DIR/$store_name"
	if [ ! -e $target_DIR/${file_name}_input_layer0.npy ]; then
		python /home/yuanw3/taskonomy/taskbank/tools/run_img_task.py --task $task --img $imgfile --store "$target_DIR/$store_name" --store-all-rep
	fi
	((i=i+1))
done </user_data/yuanw3/project_outputs/NSD/output/coco_ID_of_repeats_subj01.txt


# #for imgfile in $(ls -1 $STIMULI_DIR/* | sort -r); do
# for imgfile in $STIMULI_DIR/*; do
# 	store_name=$(basename $imgfile)
# 	file_name="${store_name%.*}"

# 	echo "processing $file_name for task $task"
# 	tmp="$(cut -d'.' -f1 <<<"$imgfile")"
# 	id="$(cut -d'/' -f7 <<<"$tmp")"
# 	printf -v old_name "COCO_train2014_%012d_input_layer0.npy" $id

# 	if [ ! -e $target_DIR/${file_name}_input_layer0.npy ] && [ ! -e $target_DIR/$old_name ]; then
# 		python /home/yuanw3/taskonomy/taskbank/tools/run_img_task.py --task $task --img $imgfile --store "$target_DIR/$store_name" --store-all-rep
# 	fi

# done