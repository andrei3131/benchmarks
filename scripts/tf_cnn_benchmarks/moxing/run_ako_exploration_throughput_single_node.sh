#!/usr/bin/env bash

set -x

cd /home/work/user-job-dir/tf_cnn_benchmarks/

# mkdir /cache/checkpoints/
mkdir /cache/train_dir/

# python moxing/prepare_input.py

KUNGFU_STRATEGY_1="partial_exchange"

KUNGFU_STRATEGY_2="partial_exchange_accumulation"

KUNGFU_STRATEGY_3="partial_exchange_accumulation_avg_peers"

KUNGFU_STRATEGY_4="partial_exchange_accumulation_avg_window"

for KUNGFU_STRATEGY in ${KUNGFU_STRATEGY_1} ${KUNGFU_STRATEGY_2} ${KUNGFU_STRATEGY_3} ${KUNGFU_STRATEGY_4}
do
    for fraction in 0.1 0.2 0.3 0.5 1
    do
        for numpeers in 1 2 4 8
        do
        # Train model using Ako
        echo "[BEGIN KEY] train-log-${KUNGFU_STRATEGY}-fraction-${fraction}-peers-${numpeers}"
        kungfu-prun -np ${numpeers} -H 127.0.0.1:${numpeers} -timeout 1000000s \
        python tf_cnn_benchmarks.py --model=resnet50 --data_name=imagenet --train_dir=/cache/train_dir \
        --num_batches=1000 --eval=False --forward_only=False --print_training_accuracy=True --tf_random_seed=123456789 \
        --num_gpus=1 --gpu_thread_mode=gpu_private --num_warmup_batches=20 --batch_size=150 \
        --piecewise_learning_rate_schedule="0.1;80;0.01;120;0.001" --momentum=0.9 --weight_decay=0.0001 \
        --optimizer=momentum --variable_update=kungfu --staged_vars=False --kungfu_strategy=${KUNGFU_STRATEGY}  --partial_exchange_fraction=${fraction} \
        --use_datasets=True --distortions=False --fuse_decode_and_crop=True \
        --resize_method=bilinear --display_every=100 --checkpoint_every_n_epochs=False --checkpoint_interval=1 \
        --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True --winograd_nonfused=True 
        echo "[END KEY] train-log-${KUNGFU_STRATEGY}-fraction-${fraction}-peers-${numpeers}"
        done
    done
done 

for numpeers in 1 2 4 8
do
    # Get throughput for KungFu Parallel SGD
    echo "[BEGIN KEY] train-log-parallel-${numpeers}"
    kungfu-prun -np ${numpeers} -H 127.0.0.1:${numpeers} -timeout 1000000s \
    python tf_cnn_benchmarks.py --model=resnet50 --data_name=imagenet --train_dir=/cache/train_dir \
    --num_batches=1000 --eval=False --forward_only=False --print_training_accuracy=True --tf_random_seed=123456789 \
    --num_gpus=1 --gpu_thread_mode=gpu_private --num_warmup_batches=20 --batch_size=150 \
    --piecewise_learning_rate_schedule="0.1;80;0.01;120;0.001" --momentum=0.9 --weight_decay=0.0001 \
    --optimizer=momentum --variable_update=kungfu --staged_vars=False --kungfu_strategy=cpu_all_reduce \
    --use_datasets=True --distortions=False --fuse_decode_and_crop=True \
    --resize_method=bilinear --display_every=100 --checkpoint_every_n_epochs=False --checkpoint_interval=1 \
    --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True --winograd_nonfused=True 
    echo "[END KEY] train-log-parallel-${numpeers}"
done

for numpeers in 1 2 4 8
do
    echo "[BEGIN KEY] train-log-replicated-${numpeers}"
    python tf_cnn_benchmarks.py --model=resnet50 --data_name=imagenet --train_dir=/cache/train_dir \
    --num_batches=1000 --eval=False --forward_only=False --print_training_accuracy=True --tf_random_seed=123456789 \
    --num_gpus=${numpeers} --gpu_thread_mode=gpu_private --num_warmup_batches=20 --batch_size=150 \
    --piecewise_learning_rate_schedule="0.1;80;0.01;120;0.001" --momentum=0.9 --weight_decay=0.0001 \
    --optimizer=momentum --variable_update=replicated --staged_vars=False \
    --use_datasets=True --distortions=False --fuse_decode_and_crop=True \
    --resize_method=bilinear --display_every=100 --checkpoint_every_n_epochs=False --checkpoint_interval=1 \
    --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True --winograd_nonfused=True 
    echo "[END KEY] train-log-replicated-${numpeers}"
done

python moxing/prepare_output.py