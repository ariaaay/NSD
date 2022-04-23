
MODELS="
clip \
resnet50_bottleneck \
clip_visual_resnet \
clip_text \
bert_layer_13"

OUTPUT_DIR="/user_data/yuanw3/project_outputs/NSD/output/encoding_results"
# OUTPUT_DIR="./output/encoding_results"


for subj in {1..8}; do
    for model in $MODELS; do
        FILE=${OUTPUT_DIR}/subj$subj/corr_${model}_whole_brain.p
        if test -f "$FILE"; then
            echo "Done"
        else
            echo "NEED $FILE"
        fi
    done

    FILE=${OUTPUT_DIR}/subj$subj/corr_clip_visual_resnet_resnet50_bottleneck_whole_brain.p
    if test -f "$FILE"; then
        echo "Done"
    else
        FILE=${OUTPUT_DIR}/subj$subj/corr_resnet50_bottleneck_clip_visual_resnet_whole_brain.p
        if test -f "$FILE"; then
            echo "Done"
        else
            echo "NEED $FILE"
        fi
    fi

    FILE=${OUTPUT_DIR}/subj$subj/corr_clip_text_bert_layer_13_whole_brain.p
    if test -f "$FILE"; then
        echo "Done"
    else
        FILE=${OUTPUT_DIR}/subj$subj/corr_bert_layer_13_clip_text_whole_brain.p
        if test -f "$FILE"; then
            echo "Done"
        else
            echo "NEED $FILE"
        fi
    fi
done




