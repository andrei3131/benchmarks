#!/usr/bin/env bash

set -x

cd /home/work/user-job-dir/tf_cnn_benchmarks/

# mkdir /cache/checkpoints/
mkdir /cache/train_dir/

# python moxing/prepare_input.py

for numpeers in 2 4 8 12 14 16
do
    # Get throughput for KungFu Parallel SGD
    echo "[BEGIN KEY] train-log-parallel-${numpeers}"
    kungfu-prun -np ${numpeers} -H 169.254.128.207:$((numpeers/2)),169.254.128.185:$((numpeers/2)) -nic ib0 -timeout 1000000s \
    python tf_cnn_benchmarks.py --model=resnet50 --data_name=imagenet --num_gpus=1\
    --variable_update=kungfu --kungfu_strategy=cpu_all_reduce \
    --num_batches=500 --eval=False --checkpoint_every_n_epochs=False \
    --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True
    echo "[END KEY] train-log-parallel-${numpeers}"
done


KUNGFU_STRATEGY_1="partial_exchange"

KUNGFU_STRATEGY_2="partial_exchange_accumulation"

KUNGFU_STRATEGY_3="partial_exchange_accumulation_avg_peers"

KUNGFU_STRATEGY_4="partial_exchange_accumulation_avg_window"

for KUNGFU_STRATEGY in ${KUNGFU_STRATEGY_4}
do
    for fraction in 0.2 0.3 0.5 1
    do
        for numpeers in 2 4 8 12 14 16
        do
        # Train model using Ako
        echo "[BEGIN KEY] train-log-${KUNGFU_STRATEGY}-fraction-${fraction}-peers-${numpeers}"
        kungfu-prun -np ${numpeers} -H 169.254.128.207:$((numpeers/2)),169.254.128.185:$((numpeers/2)) -nic ib0 -timeout 1000000s \
        python tf_cnn_benchmarks.py --model=resnet50 --data_name=imagenet --num_gpus=1\
        --variable_update=kungfu --kungfu_strategy=${KUNGFU_STRATEGY} --partial_exchange_fraction=${fraction} \
        --num_batches=500 --eval=False --checkpoint_every_n_epochs=False \
        --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True
        echo "[END KEY] train-log-${KUNGFU_STRATEGY}-fraction-${fraction}-peers-${numpeers}"
        done
    done
done 

python moxing/prepare_output.py