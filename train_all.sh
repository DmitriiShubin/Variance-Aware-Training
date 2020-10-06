#!/usr/bin/env bash
for FOLD in 0 1 2
do
    python main.py --start_fold $FOLD --gpu 0,1,2,3,4,5,6 --batch_size 128 --lr 0.001
done



