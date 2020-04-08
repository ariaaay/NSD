source venv/bin/activate

# TASKS="autoencoder \
# class_1000 \
# class_places \
# colorization \
# curvature \
TASKS="
 denoise \
 edge2d \
 edge3d \
 inpainting_whole \
 keypoint2d \
 keypoint3d \
 reshade \
 rgb2depth \
 rgb2mist \
 rgb2sfnorm \
 room_layout \
 segment25d \
 segment2d \
 segmentsemantic \
 vanishing_point"

#  jigsaw \


for task in $TASKS; do
  echo "running taskonomy $task task on subject 1"
  python code/run_modeling.py --model taskrepr_$task --subj 1  --test --permute_y
done
