#!/usr/bin/env bash
#Train models for unet
#for FOLD in 0 1 2 3 4
#do
#
#done

python main.py --start_fold 4 --n_epochs 100 --batch_size 32 --lr 1e-4 --gpu 4,5,6 --model adv_fpn &
python main.py --start_fold 3 --n_epochs 100 --batch_size 32 --lr 1e-4 --gpu 4,5,6 --model adv_fpn &


