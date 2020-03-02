TASKS="class_1000 \
 class_places \
 edge2d \
 edge3d"

#TASKS="segment25d \
# segment2d \
# reshade \
# rgb2mist \
# rgb2sfnorm \
# colorization \
# jigsaw"

#TASKS="curvature \
#autoencoder \
#denoise \
#inpainting_whole
#keypoint2d \
#keypoint3d \
#rgb2depth \
#room_layout \
#segmentsemantic \
#vanishing_point \
#"

for task in $TASKS; do
 sbatch ~/NSD/scripts/generate_taskonomy_features.sh $task
done