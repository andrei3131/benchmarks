#!/bin/bash

for g in 1; do
for lr in 0.001; do # 0.05 0.1 0.2 0.4 0.8 1.6 3.2; do
for b in 1024 512 256 128 64 32 16 8 4 2 1; do
   s=`python helpers/learningratebasicschedule-lenet.py $lr`
   # echo $b $g $s
   ./train-lenet.sh $b $g $s $lr
done
done
done

echo "Bye."
exit 0
