#!/bin/bash

# 
# Modifying the learning rate...
# We run with learning rate 0.1.
#
s="0.1;30;0.01;60;0.001;80;0.0001"
for g in 1; do
for b in 16; do
   ./train-resnet-50-dbg.sh $b $g $s
done
done
echo "Bye."
exit 0
