#!/bin/sh

source venv/bin/activate


TASKS="vanishing_point \
room_layout \
class_places \
edge3d \
class_1000 \
inpainting_whole \
segmentsemantic"


#TASKS="keypoint3d \
#segment25d \
#edge2d \
#keypoint2d


#rgb2sfnorm \
#segment2d \
#reshade \
#curvature \
#autoencoder \
#denoise \
#rgb2depth \
#colorization \
#rgb2mist \

#jigsaw

subj=$1

for task in $TASKS; do
  echo "running taskonomy $task task on subject $subj"
  python code/run_modeling.py --model taskrepr_$task --subj $subj --fix_testing --notest
done
