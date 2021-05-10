#!/usr/bin/env bash

python main.py --gpu 7 --experiment './experiments/detection/baseline/config_RSNA_2.yml'
python main.py --gpu 7 --experiment './experiments/detection/baseline/config_RSNA_4.yml'
python main.py --gpu 7 --experiment './experiments/detection/baseline/config_RSNA_8.yml'
#python main.py --gpu 7 --experiment './experiments/detection/baseline/config_RSNA_UB.yml'



