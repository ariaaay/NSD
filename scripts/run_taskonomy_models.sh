TASKS="class_1000 \
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
 room_layout \
 segmentsemantic \
 vanishing_point \
 rgb2mist \
 colorization"

#jigsaw"



for task in $TASKS; do
 sbatch ~/NSD/scripts/run_encoding_models_on_taskonomy.sh $task
done