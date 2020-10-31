#!/usr/bin/env bash
#Train models for unet
#for FOLD in 0 1 2 3 4
#do
#
#done

python main.py --start_fold 0 --n_epochs 100 --batch_size 32 --lr 1e-4 --gpu 1,0 --model fpn &
python main.py --start_fold 1 --n_epochs 100 --batch_size 32 --lr 1e-4 --gpu 2,3 --model fpn &
python main.py --start_fold 0 --n_epochs 100 --batch_size 32 --lr 1e-4 --gpu 4,5,6 --model adv_fpn &



