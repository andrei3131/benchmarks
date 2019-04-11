#!/bin/bash

for g in 1; do
for r in 0.001; do
for b in 1024 512 256 128 64 32 16 8 4; do
   version=`python helpers/findcheckpoint.py results/v17/b-${b}-g-${g}-r-${r}-ps.out`
   printf "Test results from batch size %4d on %d GPU devices (v-%s)\n" $b $g $version
   ./test-lenet.sh $version > results/v17/b-${b}-g-${g}-r-${r}-ps-test.out 2>&1
done
done
done
echo "Bye."
exit 0
