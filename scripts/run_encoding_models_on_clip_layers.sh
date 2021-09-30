modality=$1
for layer in {0..6}; do
  echo "running "${modality}_layer_${layer}" on subject 1"
  # python code/run_modeling.py --model "${modality}_layer_${layer}" --subj 1 --fix_testing
  python code/run_modeling.py --model "visual_layer_resnet${layer}" --subj 1 --fix_testing
done