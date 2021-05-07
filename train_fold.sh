#!/usr/bin/env bash
python main.py --gpu 2 --experiment './experiments/detection/baseline/config_RSNA_8_1.yml' &
python main.py --gpu 3 --experiment './experiments/detection/baseline/config_RSNA_8_2.yml' &
python main.py --gpu 4 --experiment './experiments/detection/baseline/config_RSNA_8_3.yml' &
python main.py --gpu 5 --experiment './experiments/detection/baseline/config_RSNA_8_4.yml' &
python main.py --gpu 6 --experiment './experiments/detection/baseline/config_RSNA_8_5.yml' &
python main.py --gpu 7 --experiment './experiments/detection/baseline/config_RSNA_8_6.yml' &


