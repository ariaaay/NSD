# python code/extract_clip_features.py --subj $subj

# python code/run_modeling.py --model "clip_object" --subj $subj --fix_testing
# python code/run_modeling.py --model "clip_top1_object" --subj $subj --fix_testing --subset_data "person"
# python code/run_modeling.py --model "clip" --subj 1 --fix_testing 
# python code/run_modeling.py --model "supcat" --subj $subj --fix_testing

# for layer in {0..11}; do
#  echo "running "visual_layer_${layer}" on subject ${subj}"
#  python code/run_modeling.py --model "visual_layer_${layer}" --subj $subj --fix_testing
# done

# for layer in {0..7}; do
#  echo "running "visual_layer_resnet${layer}" on subject ${subj}"
#  python code/run_modeling.py --model "visual_layer_resnet${layer}" --subj $subj --fix_testing
# done
# python code/run_modeling.py --model "clip_text" --subj $subj --fix_testing
# MODELS="
# clip \
# resnet50_bottleneck \
# clip_visual_resnet
# clip_text \
# bert_layer_13"

# for subj in {1..8}; do
#     for model in $MODELS; do
#         FILE=/user_data/yuanw3/project_outputs/NSD/output/encoding_results/subj$subj/corr_${model}_whole_brain.p
#         if test -f "$FILE"; then
#             echo "$FILE exists."
#         else
#             python code/run_modeling.py --model $model --subj $subj --fix_testing
#         fi
#     done

#     FILE=/user_data/yuanw3/project_outputs/NSD/output/encoding_results/subj$subj/corr_clip_visual_resnet_resnet50_bottleneck_whole_brain.p
#     if test -f "$FILE"; then
#         echo "$FILE exists."
#     else
#         FILE=/user_data/yuanw3/project_outputs/NSD/output/encoding_results/subj$subj/corr_resnet50_bottleneck_clip_visual_resnet_whole_brain.p
#         if test -f "$FILE"; then
#             echo "$FILE exists."
#         else
#             python code/run_modeling.py --model "clip_visual_resnet" "resnet50_bottleneck" --subj $subj --fix_testing
#         fi
#     fi

#     FILE=/user_data/yuanw3/project_outputs/NSD/output/encoding_results/subj$subj/corr_clip_text_bert_layer_13_whole_brain.p
#     if test -f "$FILE"; then
#         echo "$FILE exists."
#     else
#         FILE=/user_data/yuanw3/project_outputs/NSD/output/encoding_results/subj$subj/corr_bert_layer_13_clip_text_whole_brain.p
#         if test -f "$FILE"; then
#             echo "$FILE exists."
#         else
#             python code/run_modeling.py --model "clip_text" "bert_layer_13" --subj $subj --fix_testing
#         fi
#     fi

#     FILE=/user_data/yuanw3/project_outputs/NSD/output/encoding_results/subj$subj/corr_clip_resnet50_bottleneck_whole_brain.p
#     if test -f "$FILE"; then
#         echo "$FILE exists."
#     else
#         FILE=/user_data/yuanw3/project_outputs/NSD/output/encoding_results/subj$subj/corr_resnet50_bottleneck_clip_whole_brain.p
#         if test -f "$FILE"; then
#             echo "$FILE exists."
#         else
#             python code/run_modeling.py --model "clip" "resnet50_bottleneck" --subj $subj --fix_testing
#         fi
#     fi
# done
# python code/run_modeling.py --model "clip" --subj $subj --fix_testing
# python code/run_modeling.py --model "bert_layer_13" "clip" --subj $subj --fix_testing
# python code/run_modeling.py --model "bert_layer_13" --subj $subj --fix_testing
# python code/run_modeling.py --model "resnet50_bottleneck" --subj $subj --fix_testing
# python code/run_modeling.py --model "resnet50_bottleneck" "clip" --subj $subj --fix_testing


for subj in 1; do
    echo $subj
    # python code/run_modeling.py --model "clip_visual_resnet" --subj $subj --fix_testing --test
    # python code/run_modeling.py --model "clip" "clip_text" --subj $subj --fix_testing --test
    # python code/run_modeling.py --model "resnet50_bottleneck" --subj $subj --fix_testing --test
    # python code/run_modeling.py --model "clip" --subj $subj --fix_testing --test
    # python code/run_modeling.py --model "clip_text" --subj $subj --fix_testing --test
    python code/run_modeling.py --model "YFCC_clip" --subj $subj --fix_testing
    python code/run_modeling.py --model "YFCC_simclr" --subj $subj --fix_testing
    python code/run_modeling.py --model "YFCC_simclr" "YFCC_clip" --subj $subj --fix_testing
    python code/run_modeling.py --model "YFCC_blip" --subj $subj --fix_testing
    python code/run_modeling.py --model "YFCC_simclr" "resnet50_bottleneck" --subj $subj --fix_testing


done