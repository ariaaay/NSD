#!/bin/sh
cd /home/yuanw3/taskonomy/taskbank
source taskvenv/bin/activate
set -eu

STIMULI_DIR="/lab_data/tarrlab/common/datasets/NSD_images"
OUT_DIR="/lab_data/tarrlab/yuanw3/taskonomy_features/genStimuli/"

TASKS="autoencoder \
 denoise \
 segment25d \
 segment2d \
 curvature \
 class_1000 \
 class_places \
 edge2d \
 edge3d \
 keypoint2d \
 keypoint3d \
 reshade \
 rgb2depth \
 rgb2mist \
 rgb2sfnorm \
 colorization \
 room_layout \
 segmentsemantic \
 vanishing_point \
 jigsaw \
 inpainting_whole"


#n=0

#for imgfile in $(ls -1 $STIMULI_DIR$DIR/* | sort -r); do
for imgfile in $STIMULI_DIR/*; do
#	n=$((n + 1))
	for task in $TASKS; do
		  store_name=$(basename $imgfile)
		  target_DIR=${OUT_DIR}${task}

		  if ! [ -e $target_DIR ]; then
		    mkdir $target_DIR
		  fi

      echo "processing $imgfile for task $task"
      tmp="$(cut -d'.' -f1 <<<"$imgfile")"
      id="$(cut -d'/' -f7 <<<"$tmp")"
#      echo $id
      printf -v old_name "COCO_train2014_%012d.jpg" $id
#      echo $old_name

      if [ ! -e $target_DIR/$store_name ] && [ ! -e $target_DIR/$old_name ]; then
			  python /home/yuanw3/taskonomy/taskbank/tools/run_img_task.py --task $task --img $imgfile --store "$target_DIR/$store_name" --store-rep
		  fi
	done
done

#echo $n
