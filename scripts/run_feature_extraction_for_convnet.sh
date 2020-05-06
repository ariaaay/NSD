LAYERS="conv2 \
conv3 \
conv4 \
conv5 \
fc6"

for layer in $LAYERS; do
 sbatch ~/NSD/scripts/feature_extraction.sh $layer
done