#!/usr/bin/env bash

set -x

cd /home/work/user-job-dir/benchmarks-fresh/scripts/tf_cnn_benchmarks/

# python moxing/prepare_input.py


# export KUNGFU_CONFIG_ENABLE_MONITORING=true
# export KUNGFU_CONFIG_MONITORING_PERIOD=30s

#-np 16 -H 169.254.128.207:8,169.254.128.185:8 -nic ib0

# This was used to run with when data is sharded
# echo "[BEGIN TRAINING KEY] training-parallel"
    # python tf_cnn_benchmarks.py --model=resnet50 \
    # --data_name=imagenet \
    # --data_dir=/cache/data_dir \
    # --num_epochs=11.25 \
    # --eval=False \
    # --forward_only=False \
    # --print_training_accuracy=True \
    # --num_gpus=1 \
    # --num_warmup_batches=20 \
    # --batch_size=64 \
    # --momentum=0.9 \
    # --weight_decay=0.0001 \
    # --staged_vars=False \
    # --optimizer=p2p_averaging \
    # --variable_update=kungfu \
    # --kungfu_strategy=none \
    # --model_averaging_device=cpu \
    # --request_mode=sync \
    # --peer_selection_strategy=roundrobin \
    # --piecewise_learning_rate_schedule="0.1;3.75;0.01;7.5;0.001" \
    # --use_datasets=True \
    # --distortions=False \
    # --fuse_decode_and_crop=True \
    # --resize_method=bilinear \
    # --display_every=100 \
    # --checkpoint_every_n_epochs=True \
    # --checkpoint_interval=0.125 \
    # --checkpoint_directory=/cache/checkpoints-parallel \
    # --data_format=NCHW \
    # --batchnorm_persistent=True \
    # --use_tf_layers=True \
    # --winograd_nonfused=True 

# echo "[END TRAINING KEY] training-parallel"

# kungfu-prun -np 8 -H 127.0.0.1:8 -timeout 1000000s \
#     python tf_cnn_benchmarks.py --model=resnet50 \
#     --data_name=imagenet \
#     --data_dir=/cache/data_dir \
#     --num_epochs=11.25 \
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
#     --piecewise_learning_rate_schedule="0.1;3.75;0.01;7.5;0.001" \
#     --variable_update=kungfu \
#     --piecewise_partial_exchange_schedule="0:1,1.25:0.1,3.75:1,5:0.1,7.5:1,8.75:0.1" \
#     --kungfu_strategy=partial_exchange_group_all_reduce_with_schedule \
#     --use_datasets=True \
#     --distortions=False \
#     --fuse_decode_and_crop=True \
#     --resize_method=bilinear \
#     --display_every=100 \
#     --checkpoint_every_n_epochs=True \
#     --checkpoint_interval=0.125 \
#     --checkpoint_directory=/cache/checkpoints-parallel \
#     --data_format=NCHW \
#     --batchnorm_persistent=True \
#     --use_tf_layers=True \
#     --winograd_nonfused=True 


#IMAGENET_NUM_TRAIN_IMAGES=1276163
# CIFAR10_NUM_TRAIN_IMAGES=50000

# # This was used to run LBR
# echo "[BEGIN TRAINING KEY] training-parallel"
# N_PEERS=8
# kungfu-prun -np ${N_PEERS} -H 127.0.0.1:${N_PEERS} -timeout 1000000s \
# python tf_cnn_benchmarks.py --model=resnet50 \
#     --data_name=imagenet \
#     --data_dir=/cache/data_dir \
#     --num_epochs=11.25 \
#     --eval=False \
#     --forward_only=False \
#     --print_training_accuracy=True \
#     --num_gpus=1 \
#     --num_warmup_batches=20 \
#     --batch_size=64 \
#     --momentum=0.9 \
#     --weight_decay=0.0001 \
#     --staged_vars=False \
#     --optimizer=hybrid_p2p_averaging \
#     --variable_update=kungfu \
#     --kungfu_strategy=hybrid \
#     --model_averaging_device=gpu \
#     --request_mode=sync \
#     --shard_size=$(( ${IMAGENET_NUM_TRAIN_IMAGES}/${N_PEERS} )) \
#     --hybrid_model_averaging_schedule="0:p2p,3.75:kf" \
#     --hybrid_all_reduce_interval=1 \
#     --peer_selection_strategy=roundrobin \
#     --piecewise_learning_rate_schedule="0.1;3.75;0.01;7.5;0.001" \
#     --use_datasets=True \
#     --distortions=False \
#     --fuse_decode_and_crop=True \
#     --resize_method=bilinear \
#     --display_every=100 \
#     --checkpoint_every_n_epochs=True \
#     --checkpoint_interval=0.125 \
#     --checkpoint_directory=/cache
#     /checkpoints-parallel \
#     --data_format=NCHW \
#     --batchnorm_persistent=True \
#     --use_tf_layers=True \
#     --winograd_nonfused=True 

# echo "[END TRAINING KEY] training-parallel"


# export KUNGFU_CONFIG_ENABLE_MONITORING=true
# export KUNGFU_CONFIG_MONITORING_PERIOD=30s

echo "[BEGIN TRAINING KEY] training-parallel"

# export KUNGFU_CONFIG_ENABLE_LATENCY_MONITORING=true

## FOR VGG CURRENT RUN ON HUAWEI IS 
# Careful to prepare input
# /KungFu/bin/kungfu-huawei-launcher  
# --data_dir=/cache/data_dir \
kungfu-prun -np 16 -H 169.254.128.207:8,169.254.128.185:8 -nic ib0 -timeout 1000000s \
python tf_cnn_benchmarks.py --model=resnet50 \
    --data_name=imagenet \
    --num_epochs=30 \
    --eval=False \
    --forward_only=False \
    --print_training_accuracy=True \
    --num_gpus=1 \
    --num_warmup_batches=20 \
    --batch_size=64\
    --momentum=0.9 \
    --weight_decay=0.0001 \
    --staged_vars=False \
    --optimizer=momentum \
    --variable_update=kungfu \
    --kungfu_strategy=p2p_averaging \
    --model_averaging_device=gpu \
    --request_mode=sync \
    --use_datasets=True \
    --distortions=False \
    --fuse_decode_and_crop=True \
    --resize_method=bilinear \
    --display_every=100 \
    --checkpoint_every_n_epochs=True \
    --checkpoint_interval=0.125 \
    --checkpoint_directory=/cache/checkpoints-parallel \
    --data_format=NCHW \
    --batchnorm_persistent=True \
    --use_tf_layers=True \
    --winograd_nonfused=True 
echo "[END TRAINING KEY] training-parallel"


for worker in 0 1 2 3 4 5 6 7
do
echo "[BEGIN VALIDATION KEY] validation-parallel-worker-${worker}"
python tf_cnn_benchmarks.py --eval=True --forward_only=False --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir \
    --variable_update=replicated --data_format=NCHW --use_datasets=False --num_batches=50 --eval_batch_size=150 \
    --num_gpus=8 --use_tf_layers=True \
    --checkpoint_directory=/cache/checkpoints-parallel-worker-${worker}/v-000001 --checkpoint_interval=0.125 --checkpoint_every_n_epochs=True 
echo "[END VALIDATION KEY] validation-parallel-worker-${worker}"
done


#python moxing/prepare_output.py