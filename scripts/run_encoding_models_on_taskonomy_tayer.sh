#!/bin/sh

source venv/bin/activate


TASKS="class_1000 \
vanishing_point \
room_layout \
class_places \
edge2d \
edge3d \
rgb2sfnorm \
segment25d \
segment2d \
reshade \
curvature \
autoencoder \
denoise \
inpainting_whole
keypoint2d \
keypoint3d \
rgb2depth \
segmentsemantic \
colorization \
rgb2mist"

#jigsaw

sub=$1

for task in $TASKS; do
  echo "running taskonomy $task task on subject $sub"
  python code/run_modeling.py --model taskrepr_$task --subj $sub --fix_testing --notest
done
