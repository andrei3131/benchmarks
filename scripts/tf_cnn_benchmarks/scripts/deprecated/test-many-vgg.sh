#!/bin/bash

# for g in 8 4; do
# for r in 0.1; do
# for b in 1024 512 256 128 64 32 16 8; do
for g in 8; do
for r in 0.4 0.2 0.1 0.05 0.025; do
for b in 32; do 
   version=`python helpers/findcheckpoint.py results/v19/b-${b}-g-${g}-r-${r}.out`
   printf "Test results from batch size %4d on %d GPU devices (v-%s)\n" $b $g $version
   ./test-vgg.sh $version > results/v19/b-${b}-g-${g}-r-${r}-test.out 2>&1
done
done
done
echo "Bye."
exit 0
