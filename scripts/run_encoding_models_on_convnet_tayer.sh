#!/bin/sh

source venv/bin/activate

FEATURES="vgg16 \
res50
"

sub=1

for feature in $FEATURES; do
  echo "running convnet $feature on subject $sub"
  python nsd_code/run_modeling.py --model convnet_$feature --subj $sub --fix_testing --notest
done
