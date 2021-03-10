#!/usr/bin/env bash
python main.py --gpu 1 --layer_feature_maps [8,16] --wavenet_dilation [2,4,8,16,32] --model wavenet --dropout 0.1 &
python main.py --gpu 2 --layer_feature_maps [8,16,32] --wavenet_dilation [2,4,8,16,32]  --model wavenet --dropout 0.1 &
python main.py --gpu 3 --layer_feature_maps [8,16] --wavenet_dilation [2,4,8,16,32]  --model wavenet --dropout 0.2 &
python main.py --gpu 4 --layer_feature_maps [8,16,32] --wavenet_dilation [2,4,8,16,32]  --model wavenet --dropout 0.2 &








