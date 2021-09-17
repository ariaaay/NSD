modality=$1
for layer in {0..11}; do
  echo "running "${modality}_layer_${layer}" on subject 1"
  python code/run_modeling.py --model "${modality}_layer_${layer}" --subj 1 --fix_testing
done