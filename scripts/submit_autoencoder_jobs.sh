FNUM="1 \
2 \
3 \
5"

VNUM="1 \
2 \
3 \
4 \
5 \
6 \
7"

for num in $FNUM; do
 sbatch ~/NSD/scripts/train_autoencoder.sh floc-faces $num
done

for num in $VNUM; do
 sbatch ~/NSD/scripts/train_autoencoder.sh prf-visualrois $num
done

