#!/bin/bash

which python
python main.py --data datasets/zbaa_clean.csv --config wspd_3cir.json > /Users/monkeybutter/Dropbox/wspd_3cir.out &
python main.py --data datasets/zbaa_clean.csv --config wspd_3cir_press.json > /Users/monkeybutter/Dropbox/wspd_3cir_press.out &
wait
python main.py --data datasets/zbaa_clean.csv --config wspd_3cir_rh.json > /Users/monkeybutter/Dropbox/wspd_3cir_rh.out &
python main.py --data datasets/zbaa_clean.csv --config wspd_2cir.json > /Users/monkeybutter/Dropbox/wspd_2cir.out &
wait
python main.py --data datasets/zbaa_clean.csv --config wspd_2cir_press.json > /Users/monkeybutter/Dropbox/wspd_2cir_press.out &
python main.py --data datasets/zbaa_clean.csv --config wspd_2cir_rh.json > /Users/monkeybutter/Dropbox/wspd_2cir_rh.out &
wait
python main.py --data datasets/zbaa_clean.csv --config temp_3cir.json > /Users/monkeybutter/Dropbox/temp_3cir.out &
python main.py --data datasets/zbaa_clean.csv --config temp_3cir_press.json > /Users/monkeybutter/Dropbox/temp_3cir_press.out &
wait
python main.py --data datasets/zbaa_clean.csv --config temp_3cir_rh.json > /Users/monkeybutter/Dropbox/temp_3cir_rh.out &
python main.py --data datasets/zbaa_clean.csv --config temp_2cir.json > /Users/monkeybutter/Dropbox/temp_2cir.out &
wait
python main.py --data datasets/zbaa_clean.csv --config temp_2cir_press.json > /Users/monkeybutter/Dropbox/temp_2cir_press.out &
python main.py --data datasets/zbaa_clean.csv --config temp_2cir_rh.json > /Users/monkeybutter/Dropbox/temp_2cir_rh.out &
wait
