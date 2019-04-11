#!/bin/bash

# for g in 8; do
# for r in 0.05 0.1 0.2 0.4 0.8 1.6 3.2; do
# for b in 1024 512 256 128 64 32 16; do
#   version=`python helpers/findcheckpoint.py results/v5/b-${b}-g-${g}-r-${r}.out`
#   printf "Test results from batch size %4d on %d GPU devices and rate %4.2f (v-%s)\n" $b $g $r $version
#   ./test.sh $version > results/v5/b-${b}-g-${g}-r-${r}-test.out 2>&1
# done
# done
# done

for g in 1; do
for r in 0.1; do
for b in 8 16 32 64 128 256 512 1024; do
   version=`python helpers/findcheckpoint.py results/v13/b-${b}-g-${g}-r-${r}-ps.out`
   printf "Test results from batch size %4d on %d GPU devices and rate %4.2f (v-%s)\n" $b $g $r $version
   ./test.sh $version > results/v13/b-${b}-g-${g}-r-${r}-ps-test.out 2>&1
done
done
done
echo "Bye."
exit 0
