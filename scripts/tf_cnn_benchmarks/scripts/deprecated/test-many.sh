#!/bin/bash

for g in 8 4 2 1; do
for b in 1024 512 256 128 64 32 16; do
   version=`python helpers/findcheckpoint.py results/v4/b-${b}-g-${g}.out`
   printf "Test results from batch size %4d on %d GPU devices (v-%s)\n" $b $g $version
   ./test.sh $version > results/v4/b-${b}-g-${g}-test.out 2>&1
done
done
echo "Bye."
exit 0
