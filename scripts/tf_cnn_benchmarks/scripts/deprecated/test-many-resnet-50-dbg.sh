#!/bin/bash

for g in 8; do
for b in 32; do
   version=`python helpers/findcheckpoint.py results/v12/b-${b}-g-${g}-same-resnet.out`
   printf "Test results from batch size %4d on %d GPU devices (v-%s)\n" $b $g $version
   ./test-resnet-50.sh $version # > results/v12/b-${b}-g-${g}-debug.out 2>&1
done
done
echo "Bye."
exit 0
