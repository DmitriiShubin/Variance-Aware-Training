#!/usr/bin/env bash
python main.py --gpu 4 --experiment './experiments/detection/baseline/config_RSNA_2.yml' &
python main.py --gpu 5 --experiment './experiments/detection/baseline/config_RSNA_2_1.yml' &
python main.py --gpu 6 --experiment './experiments/detection/baseline/config_RSNA_2_2.yml' &
python main.py --gpu 7 --experiment './experiments/detection/baseline/config_RSNA_2_3.yml' &


