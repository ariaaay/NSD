#TASKS="class_1000 \
# class_places \
# edge2d \
# edge3d"

TASKS="rgb2mist \
 colorization \
 jigsaw"

#TASKS="rgb2sfnorm \
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
#room_layout \
#segmentsemantic \
#vanishing_point \
#"

for task in $TASKS; do
 sbatch ~/NSD/scripts/generate_taskonomy_features.sh $task
done