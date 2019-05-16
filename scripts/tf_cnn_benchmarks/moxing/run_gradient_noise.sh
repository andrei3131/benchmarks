#!/usr/bin/env bash

set -x

cd /home/work/user-job-dir/tf_cnn_benchmarks/

# Use this when placing code in andrei/code
# cd /home/work/user-job-dir/code/kungfu/
# ./configure --build-tensorflow-ops
# ldconfig /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs/
# env KUNGFU_USE_NCCL=0 pip install --no-index -U . 
# ldconfig
# ./scripts/go-install.sh
# cd /home/work/user-job-dir/code/benchmarks/scripts/tf_cnn_benchmarks

python moxing/prepare_input.py

# Use 2>&1 | to redirect stderr to tee file


# Random seed has been removed from experiments


echo "[BEGIN TRAINING KEY] training-parallel-gradient-noise"
kungfu-prun  -np 8 -H 127.0.0.1:8 -timeout 1000000s \
    python tf_cnn_benchmarks.py --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir --train_dir=/cache/train_dir \
    --num_epochs=90 --eval=False --forward_only=False --print_training_accuracy=True \
    --num_gpus=1 --num_warmup_batches=20 --batch_size=150 \
    --momentum=0.9 --weight_decay=0.0001 \
    --optimizer=momentum --variable_update=kungfu --staged_vars=False --kungfu_strategy=cpu_all_reduce_noise \
    --use_datasets=True --distortions=False --fuse_decode_and_crop=True \
    --resize_method=bilinear --display_every=100 --checkpoint_every_n_epochs=True --checkpoint_interval=0.125 \
    --checkpoint_directory=/cache/checkpoints-parallel \
    --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True --winograd_nonfused=True \
    --save_summaries_steps=1 --summary_verbosity=1
echo "[END TRAINING KEY] training-parallel-gradient-noise"



kungfu-prun  -np 4 -H 127.0.0.1:4 -timeout 1000000s \
    python3 tf_cnn_benchmarks.py --model=resnet32 --data_name=cifar10 --data_dir=/data/cifar-10/cifar-10-batches-py/ \
    --num_epochs=20 --eval=False --forward_only=False --print_training_accuracy=True \
    --num_gpus=1 --num_warmup_batches=20 --batch_size=64 \
    --momentum=0.9 --weight_decay=0.0001 \
    --optimizer=momentum --variable_update=kungfu --staged_vars=False --kungfu_strategy=cpu_all_reduce_noise \
    --use_datasets=True --distortions=False --fuse_decode_and_crop=True \
    --resize_method=bilinear --display_every=1 --checkpoint_every_n_epochs=False --checkpoint_interval=0.125 \
    --checkpoint_directory=/cache/checkpoints-parallel \
    --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True --winograd_nonfused=True \
    --save_summaries_steps=1 --summary_verbosity=1


python moxing/prepare_output.py