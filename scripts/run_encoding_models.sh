# modality=$1
for layer in {1..13}; do
  echo "running "bert_layer_${layer}" on subject 1"
  # python code/run_modeling.py --model "${modality}_layer_${layer}" --subj 1 --fix_testing
  python code/run_modeling.py --model "bert_layer_${layer}" --subj 1 --fix_testing
done

# python code/run_modeling.py --model "clip_object_caption_overlap" --subj 1 --fix_testing
# python code/run_modeling.py --model "clip_top1_object" --subj 1 --fix_testing