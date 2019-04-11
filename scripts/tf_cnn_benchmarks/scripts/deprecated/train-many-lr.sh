#!/bin/bash

# 
# Modifying the learning rate...
# We run with learning rate 0.1.
# If 512 0.1, then 8192 / 512 = 16 so x0.1 = 1.6
# 
# for g in 8; do
# for lr in 0.05 0.1 0.2 0.4 0.8 1.6 3.2; do
# for b in 1024 512 256 128 64 32 16; do
#    s=`python helpers/learningratebasicschedule.py $lr`
#    # echo $b $g $s
#    ./train-lr.sh $b $g $s $lr
# done
# done
# done

# for g in 1; do
# for lr in 0.1; do # 0.05 0.1 0.2 0.4 0.8 1.6 3.2; do
# for b in 1024 512 256 128 64 32 16 8 4 2 1; do
for g in 4; do
for lr in 0.1; do # 0.05 0.1 0.2 0.4 0.8 1.6 3.2; do
for b in 256; do
   s=`python helpers/learningratebasicschedule.py $lr`
   # echo $b $g $s
   ./train-lr.sh $b $g $s $lr
done
done
done

echo "Bye."
exit 0
