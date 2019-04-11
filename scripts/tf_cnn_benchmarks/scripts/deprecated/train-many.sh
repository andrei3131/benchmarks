#!/bin/bash

# 
# Modifying the learning rate...
# We run with learning rate 0.1.
# 
# for g in 4 2 1; do
# for b in 1024 512 256 128 64 32 16; do
for g in 4; do
for b in 256; do
   ./train.sh $b $g
done
done
echo "Bye."
exit 0
