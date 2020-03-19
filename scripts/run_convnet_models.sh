FEATURES="vgg16 \
res50
"

for feature in $FEATURES; do
 sbatch ~/NSD/scripts/run_encoding_models_on_convnet.sh $feature
done