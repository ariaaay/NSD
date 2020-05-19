FEATURES="alexnet_conv1 \
alexnet_conv2 \
alexnet_conv3 \
alexnet_conv4 \
alexnet_conv5 \
alexnet_fc6 \
alexnet_fc7
"

for feature in $FEATURES; do
 sbatch ~/NSD/scripts/run_encoding_models_on_convnet.sh $feature
done