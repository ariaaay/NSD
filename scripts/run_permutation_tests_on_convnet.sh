source venv/bin/activate

 MODELS="res50"

for model in $MODELS; do
  echo "running convnet $model on subject 1"
  python nsd_code/run_modeling.py --model convnet_$model --subj 1  --test --permute_y
done
