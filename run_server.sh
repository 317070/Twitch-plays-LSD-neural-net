#!/bin/bash

rm logfile

while true; do
python train.py >> logfile 2>&1
killall -s 9 avconv
killall -s 9 python
sleep 1
done
