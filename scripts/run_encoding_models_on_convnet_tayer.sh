#!/bin/sh

source venv/bin/activate

FEATURES="alexnet_conv1 \
alexnet_conv2 \
alexnet_conv3 \
alexnet_conv4 \
alexnet_conv5 \
alexnet_fc6 \
alexnet_fc7
"


sub=1

for feature in $FEATURES; do
  echo "running convnet $feature on subject $sub"
  python code/run_modeling.py --model convnet_$feature --subj $sub --fix_testing --notest
done
