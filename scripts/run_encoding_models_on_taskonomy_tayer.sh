#!/bin/sh

source venv/bin/activate


TASKS="vanishing_point \
room_layout \
class_places \
edge2d \
edge3d"
#class_1000 \
#rgb2sfnorm \
#segment25d \
#segment2d \
#reshade \
#curvature \
#autoencoder \
#denoise \
#inpainting_whole
#keypoint2d \
#keypoint3d \
#rgb2depth \
#segmentsemantic \
#colorization \
#rgb2mist"

#jigsaw

SUBJS="2 \
5\
7"

#sub=$1
for sub in $SUBJS; do
  for task in $TASKS; do
    echo "running taskonomy $task task on subject $sub"
    python code/run_modeling.py --model taskrepr_$task --subj $sub --fix_testing --notest
  done
done
