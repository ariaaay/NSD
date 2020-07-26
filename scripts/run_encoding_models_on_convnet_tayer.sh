#!/bin/sh

source venv/bin/activate

#FEATURES="alexnet_conv1_avgpool \
#alexnet_conv2_avgpool \
#alexnet_conv3_avgpool \
#alexnet_conv4_avgpool \
#alexnet_conv5_avgpool \
#alexnet_fc6_avgpool \
#alexnet_fc7__avgpool
#"
FEATURES="res50"


sub=1

for feature in $FEATURES; do
  echo "running convnet $feature on subject $sub"
  python code/run_modeling.py --model convnet_$feature --subj $sub --fix_testing --notest
done
