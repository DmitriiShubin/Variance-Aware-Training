#!/usr/bin/env bash
for FOLD in 1 2 3 4
do
  (python main.py --start_fold $FOLD --gpu $FOLD --model vgg
  python main.py --start_fold $FOLD --gpu $FOLD --model resnet
  python main.py --start_fold $FOLD --gpu $FOLD --model resnet_xt
  ) &
done




