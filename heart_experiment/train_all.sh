#!/usr/bin/env bash
#Train models for unet
for FOLD in 0 1 2 3 4
do
  python main.py --start_fold $FOLD --n_epochs 100 --batch_size 32 --lr 1e-3 --gpu 0,1,2 --model unet
  python main.py --start_fold $FOLD --n_epochs 100 --batch_size 32 --lr 1e-3 --gpu 0,1,2 --alpha 5 --model adv_unet
  python main.py --start_fold $FOLD --n_epochs 100 --batch_size 32 --lr 1e-3 --gpu 0,1,2 --model fpn
  python main.py --start_fold $FOLD --n_epochs 100 --batch_size 32 --lr 1e-3 --gpu 0,1,2 --alpha 5 --model adv_fpn
done






