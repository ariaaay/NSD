source venv/bin/activate

 TASKS="autoencoder \
 class_1000 \
 class_places \
 colorization \
 curvature \
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


task=$1

echo "running taskonomy $task task on subject 1"
python nsd_code/run_modeling.py --model taskrepr_$task --whole_brain --subj 1  --test --permute_y

