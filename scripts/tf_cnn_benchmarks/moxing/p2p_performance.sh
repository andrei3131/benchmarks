#!/usr/bin/env bash

set -x

cd /home/work/user-job-dir/benchmarks-fresh/scripts/tf_cnn_benchmarks/

# python moxing/prepare_input.py


# export KUNGFU_CONFIG_ENABLE_MONITORING=true
# export KUNGFU_CONFIG_MONITORING_PERIOD=30s


echo "[BEGIN TRAINING KEY] training-SYNC"
# Use synthetic data to test performance
/KungFu/bin/kungfu-huawei-launcher -timeout 1000000s \
    python tf_cnn_benchmarks.py --model=resnet50 \
    --data_name=imagenet \
    --num_batches=300 \
    --eval=False \
    --forward_only=False \
    --print_training_accuracy=True \
    --num_gpus=1 \
    --num_warmup_batches=20 \
    --batch_size=64 \
    --momentum=0.9 \
    --weight_decay=0.0001 \
    --staged_vars=False \
    --optimizer=p2p_averaging \
    --variable_update=kungfu \
    --kungfu_strategy=none \
    --type_of_decentralized_synchronization=async_gpu \
    --use_datasets=True \
    --distortions=False \
    --fuse_decode_and_crop=True \
    --resize_method=bilinear \
    --display_every=100 \
    --checkpoint_every_n_epochs=False \
    --checkpoint_interval=0.125 \
    --checkpoint_directory=/cache/checkpoints-parallel \
    --data_format=NCHW \
    --batchnorm_persistent=True \
    --use_tf_layers=True \
    --winograd_nonfused=True 
echo "[END TRAINING KEY] training-SYNC"

# echo "[BEGIN TRAINING KEY] training-ASYNC"
# # Use synthetic data to test performance
# /KungFu/bin/kungfu-huawei-launcher -timeout 1000000s \
#     python tf_cnn_benchmarks.py --model=resnet50 \
#     --data_name=imagenet \
#     --num_batches=300 \
#     --eval=False \
#     --forward_only=False \
#     --print_training_accuracy=True \
#     --num_gpus=1 \
#     --num_warmup_batches=20 \
#     --batch_size=64 \
#     --momentum=0.9 \
#     --weight_decay=0.0001 \
#     --staged_vars=False \
#     --optimizer=p2p_averaging \
#     --variable_update=kungfu \
#     --kungfu_strategy=none \
#     --type_of_decentralized_synchronization=async \
#     --use_datasets=True \
#     --distortions=False \
#     --fuse_decode_and_crop=True \
#     --resize_method=bilinear \
#     --display_every=100 \
#     --checkpoint_every_n_epochs=False \
#     --checkpoint_interval=0.125 \
#     --checkpoint_directory=/cache/checkpoints-parallel \
#     --data_format=NCHW \
#     --batchnorm_persistent=True \
#     --use_tf_layers=True \
#     --winograd_nonfused=True 
# echo "[END TRAINING KEY] training-ASYNC"


# echo "[BEGIN TRAINING KEY] training-KUNGFU"
# # Use synthetic data to test performance
# /KungFu/bin/kungfu-huawei-launcher -timeout 1000000s \
#     python tf_cnn_benchmarks.py --model=resnet50 \
#     --data_name=imagenet \
#     --num_batches=300 \
#     --eval=False \
#     --forward_only=False \
#     --print_training_accuracy=True \
#     --num_gpus=1 \
#     --num_warmup_batches=20 \
#     --batch_size=64 \
#     --momentum=0.9 \
#     --weight_decay=0.0001 \
#     --staged_vars=False \
#     --optimizer=momentum \
#     --variable_update=kungfu \
#     --kungfu_strategy=cpu_all_reduce \
#     --use_datasets=True \
#     --distortions=False \
#     --fuse_decode_and_crop=True \
#     --resize_method=bilinear \
#     --display_every=100 \
#     --checkpoint_every_n_epochs=False \
#     --checkpoint_interval=0.125 \
#     --checkpoint_directory=/cache/checkpoints-parallel \
#     --data_format=NCHW \
#     --batchnorm_persistent=True \
#     --use_tf_layers=True \
#     --winograd_nonfused=True 
# echo "[END TRAINING KEY] training-KUNGFU"


# echo "[BEGIN TRAINING KEY] training-INDEPENDENT"
# # Use synthetic data to test performance
#     python tf_cnn_benchmarks.py --model=resnet50 \
#     --data_name=imagenet \
#     --num_batches=300 \
#     --eval=False \
#     --forward_only=False \
#     --print_training_accuracy=True \
#     --num_gpus=1 \
#     --num_warmup_batches=20 \
#     --batch_size=64 \
#     --momentum=0.9 \
#     --weight_decay=0.0001 \
#     --staged_vars=False \
#     --optimizer=momentum \
#     --variable_update=independent \
#     --use_datasets=True \
#     --distortions=False \
#     --fuse_decode_and_crop=True \
#     --resize_method=bilinear \
#     --display_every=100 \
#     --checkpoint_every_n_epochs=False \
#     --checkpoint_interval=0.125 \
#     --checkpoint_directory=/cache/checkpoints-parallel \
#     --data_format=NCHW \
#     --batchnorm_persistent=True \
#     --use_tf_layers=True \
#     --winograd_nonfused=True 
# echo "[END TRAINING KEY] training-INDEPENDENT"

#python moxing/prepare_output.py