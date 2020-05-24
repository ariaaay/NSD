LAYERS="conv1 \
conv2 \
conv3 \
conv4 \
conv5"
#fc6 \
#fc7"

for layer in $LAYERS; do
 sbatch ~/NSD/scripts/feature_extraction.sh $layer
done