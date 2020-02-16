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

for task in $TASKS; do
  set -eu
  sbatch ~/NSD/generate_taskonomy_feature.sh $task
done