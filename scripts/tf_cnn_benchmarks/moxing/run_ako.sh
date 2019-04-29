#!/usr/bin/env bash

set -x

cd /home/work/user-job-dir/tf_cnn_benchmarks/

mkdir /cache/checkpoints/

python moxing/prepare_input.py

# Train model using Ako
kungfu-prun -np $1 -H $2 -timeout 1000000s \
    python tf_cnn_benchmarks.py --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir --train_dir=/cache/train_dir \
    --num_batches=200 --eval=False --forward_only=False --print_training_accuracy=True --tf_random_seed=123456789 \
    --num_gpus=1 --gpu_thread_mode=gpu_private --num_warmup_batches=20 --batch_size=150 \
    --piecewise_learning_rate_schedule="0.1;80;0.01;120;0.001" --momentum=0.9 --weight_decay=0.0001 \
    --optimizer=momentum --variable_update=kungfu --staged_vars=False --kungfu_strategy=partial_exchange \
    --partial_exchange_fraction=0.1 --use_datasets=True --distortions=False --fuse_decode_and_crop=True \
    --resize_method=bilinear --display_every=100 --checkpoint_every_n_epochs=True --checkpoint_interval=1 \
    --checkpoint_directory=/cache/checkpoints \
    --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True --winograd_nonfused=True

# Evaluate the checkpoint and print the accuracy over epochs
for worker in 0 1 2 3 
do
python tf_cnn_benchmarks.py --eval=True --forward_only=False --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir \
    --variable_update=replicated --data_format=NCHW --use_datasets=False --num_epochs=1 --eval_batch_size=150 \
    --tf_random_seed=123456789 --num_gpus=8 --use_tf_layers=True \
    --checkpoint_directory=/cache/checkpoints-worker-${worker}/v-000001 --checkpoint_interval=1 --checkpoint_every_n_epochs=True 
done

python moxing/prepare_output.py