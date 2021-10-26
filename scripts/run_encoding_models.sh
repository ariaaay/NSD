
subj=1

# subset=$1
# python code/run_modeling.py --model "clip" --subj $subj --fix_testing --subset_data ${subset}
# python code/run_modeling.py --model "cat" --subj $subj --fix_testing --subset_data ${subset}
# python code/run_modeling.py --model "convnet_res50" --subj $subj --fix_testing --subset_data ${subset}

# python code/run_modeling.py --model "supcat" --subj $subj --fix_testing

# python code/extract_clip_features.py --subj $subj

# python code/run_modeling.py --model "clip_object" --subj $subj --fix_testing
# python code/run_modeling.py --model "clip_top1_object" --subj $subj --fix_testing --subset_data "person"
# python code/run_modeling.py --model "clip" --subj 1 --fix_testing 



# python code/run_modeling.py --model "resnet50_bottleneck" --subj 2 --fix_testing
# python code/run_modeling.py --model "resnet50_bottleneck" --subj 2 --fix_testing --subset_data "person"



# for layer in {0..11}; do
#  echo "running "visual_layer_${layer}" on subject ${subj}"
#  python code/run_modeling.py --model "visual_layer_${layer}" --subj $subj --fix_testing
# done

# for layer in {0..7}; do
#  echo "running "visual_layer_resnet${layer}" on subject ${subj}"
#  python code/run_modeling.py --model "visual_layer_resnet${layer}" --subj $subj --fix_testing
# done
# python code/run_modeling.py --model "clip_text" --subj $subj --fix_testing


python code/run_modeling.py --model "bert_layer_13" "clip" --subj $subj --fix_testing
