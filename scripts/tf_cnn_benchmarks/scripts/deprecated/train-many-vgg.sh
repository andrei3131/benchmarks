#!/bin/bash

for g in 8; do
for lr in 0.4 0.2 0.1 0.05 0.025; do # 0.05 0.1 0.2 0.4 0.8 1.6 3.2; do
for b in 32; do # 1024 512 256 128 64 32 16 8 4 2 1; do
   s=`python helpers/learningratebasicschedule-vgg.py $lr`
   # echo $b $g $s
   ./train-vgg.sh $b $g $s $lr
done
done
done

echo "Bye."
exit 0
